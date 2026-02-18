# -*- coding: utf-8 -*-
import os
import io
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import EsmForMaskedLM, AutoModel, EsmTokenizer, AutoTokenizer

# Project-local imports
from process_dataset import DatabaseProcessor, BatchFileDataset

# Avoid HF tokenisers forking too many workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

__all__ = [
    "get_feature",
    "collate_multiple_pt_files",
    "get_data_loader",
    "encode_pretrained_feature",
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction and Sharding
# ─────────────────────────────────────────────────────────────────────────────
def get_feature(model, dataloader, args, set_type: str, save_interval: int = 10) -> List[str]:
    """
    Encode raw inputs into token embeddings and masks, sharded as .pt files.

    Notes
    -----
    - Saves shards periodically to reduce memory pressure.
    - Returns a list of file paths to the saved shards.
    """
    subdirectory = os.path.join(
        args.input_feature_save_path, args.prot_encoder_path, args.split
    )
    os.makedirs(subdirectory, exist_ok=True)

    batch_counter = 0
    tmp_files: List[str] = []

    all_prot_embeds, all_drug_embeds = [], []
    all_prot_masks, all_drug_masks = [], []
    all_attn_maps, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for step, batch in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Encoding {set_type}"
        ):
            (
                prot_input_ids,
                prot_attention_mask,
                drug_input_ids,
                drug_attention_mask,
                atte_site,
                label,
            ) = batch

            prot_input_ids = prot_input_ids.to(args.device)
            prot_attention_mask = prot_attention_mask.to(args.device)
            drug_input_ids = drug_input_ids.to(args.device)
            drug_attention_mask = drug_attention_mask.to(args.device)

            prot_embed, drug_embed = model.encoding(
                prot_input_ids, prot_attention_mask, drug_input_ids, drug_attention_mask
            )

            all_prot_embeds.append(prot_embed.cpu())
            all_drug_embeds.append(drug_embed.cpu())
            all_prot_masks.append(prot_attention_mask.cpu())
            all_drug_masks.append(drug_attention_mask.cpu())
            all_labels.append(label.cpu())
            all_attn_maps.append(atte_site)

            flush = ((step + 1) % save_interval == 0) or ((step + 1) == len(dataloader))
            if flush:
                partial_file = os.path.join(
                    subdirectory, f"{args.dataset}_{set_type}_part_{batch_counter}.pt"
                )
                torch.save(
                    {
                        "prot": torch.cat(all_prot_embeds),
                        "drug": torch.cat(all_drug_embeds),
                        "prot_mask": torch.cat(all_prot_masks),
                        "drug_mask": torch.cat(all_drug_masks),
                        "attn_maps": all_attn_maps,  # list[str]
                        "y": torch.cat(all_labels),
                    },
                    partial_file,
                )
                tmp_files.append(partial_file)
                batch_counter += 1

                all_prot_embeds, all_drug_embeds = [], []
                all_prot_masks, all_drug_masks = [], []
                all_attn_maps, all_labels = [], []

    return tmp_files


def collate_multiple_pt_files(batch_files: Sequence[str]):
    """
    Collate function that reads a batch of .pt shard files and concatenates them.
    Returns:
        prot (T, Dp), drug (T, Dd), prot_mask (T, Lp), drug_mask (T, Ll),
        flat_attn_maps (List[str]), labels (T, 1)
    """
    all_prot, all_drug = [], []
    all_prot_mask, all_drug_mask = [], []
    all_attn_maps, all_labels = [], []

    for file_path in batch_files:
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        all_prot.append(data["prot"])
        all_drug.append(data["drug"])
        all_prot_mask.append(data["prot_mask"])
        all_drug_mask.append(data["drug_mask"])
        all_attn_maps.append(data["attn_maps"])
        all_labels.append(data["y"])

    flat_attn_maps = [item for sublist in all_attn_maps for item in sublist]

    return (
        torch.cat(all_prot, dim=0),
        torch.cat(all_drug, dim=0),
        torch.cat(all_prot_mask, dim=0),
        torch.cat(all_drug_mask, dim=0),
        flat_attn_maps,
        torch.cat(all_labels, dim=0),
    )


def get_data_loader(file_list, batch_file: int = 1, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    """
    DataLoader over shard file paths; each batch yields concatenated tensors
    via `collate_multiple_pt_files`.
    """
    if isinstance(file_list, (str, os.PathLike)):
        file_list = [file_list]
    elif not isinstance(file_list, (list, tuple)):
        raise TypeError(f"file_list must be list/tuple/str, got {type(file_list)}")

    dataset = BatchFileDataset([str(p) for p in file_list])

    return DataLoader(
        dataset,
        batch_size=batch_file,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_multiple_pt_files,
    )

def encode_pretrained_feature(args):
    """
    NEW DESIGN:
    -----------
    ❌ No pre-encoding
    ❌ No embedding cache
    ✅ Raw text -> encoder -> model (on-the-fly)

    Return:
        train_loader, valid_loader, test_loader, encoders
    """

    # --------------------------------------------------
    # Build tokenizers & encoders (ALWAYS)
    # --------------------------------------------------
    prot_tokenizer = EsmTokenizer.from_pretrained(
        args.prot_encoder_path, do_lower_case=False
    )
    drug_tokenizer = AutoTokenizer.from_pretrained(
        args.drug_encoder_path, trust_remote_code=True
    )

    prot_model = EsmForMaskedLM.from_pretrained(
        args.prot_encoder_path
    ).to(args.device)
    drug_model = AutoModel.from_pretrained(
        args.drug_encoder_path, trust_remote_code=True
    ).to(args.device)

    prot_model.eval()
    drug_model.eval()

    # --------------------------------------------------
    # Collate: raw text -> token ids
    # --------------------------------------------------
    def collate_fn_batch_encoding(batch):
        query1, query2, atte_site, scores = zip(*batch)

        prot_enc = prot_tokenizer.batch_encode_plus(
            list(query1),
            max_length=512,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        drug_enc = drug_tokenizer.batch_encode_plus(
            list(query2),
            max_length=512,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return (
            prot_enc["input_ids"],
            prot_enc["attention_mask"].bool(),
            drug_enc["input_ids"],
            drug_enc["attention_mask"].bool(),
            list(atte_site),
            torch.tensor(scores, dtype=torch.float32).view(-1, 1),
            # torch.tensor([float(s) for s in scores], dtype=torch.float32).view(-1, 1),

        )

    dp = DatabaseProcessor(args)

    # --------------------------------------------------
    # Build dataloaders (NO encoding here)
    # --------------------------------------------------
    train_loader = valid_loader = test_loader = None

    if args.mode == "train":
        train_examples = dp.get_train_examples()
        valid_examples = dp.get_val_examples()

        train_loader = DataLoader(
            train_examples,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_batch_encoding,
        )

        valid_loader = DataLoader(
            valid_examples,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_batch_encoding,
        )

    test_examples = dp.get_test_examples()
    test_loader = DataLoader(
        test_examples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_batch_encoding,
    )

    print(
        "[On-the-fly encoding] "
        f"train={len(train_examples) if train_loader else 0}, "
        f"valid={len(valid_examples) if valid_loader else 0}, "
        f"test={len(test_examples)}"
    )

    encoders = (prot_model, drug_model)

    return train_loader, valid_loader, test_loader, encoders