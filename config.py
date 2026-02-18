# -*- coding: utf-8 -*-
import argparse
import os
import torch

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f")  # allows running in notebooks without arg errors

    parser.add_argument(
        "--prot_encoder_path",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        # westlake-repl/SaProt_650M_PDB, westlake-repl/SaProt_650M_AF2 facebook/esm2_t33_650M_UR50D
        help="Name or path of the protein encoder.",
    )
    parser.add_argument(
        "--drug_encoder_path",
        type=str,
        default="HUBioDataLab/SELFormer",
        # ibm-research/materials.selfies-ted,  ibm-research/materials.smi-ted,  ibm/MoLFormer-XL-both-10pct"
        help="Name or path of the SELFIES/SMILES encoder.",
    )
    parser.add_argument(
        "--input_feature_save_path",
        type=str,
        default="data/processed_feature/",
        help="Directory for cached, pre-encoded features (.pt shards).",
    )
    parser.add_argument(
        "--agg_mode", default="mean_all_tok", type=str, help="{attention|cls|mean|mean_all_tok}"
    )
    parser.add_argument("--lambda_warmup_epochs", type=int, default=5,
                        help="Linearly ramp lambda_attn from 0 to target over these epochs.")

    parser.add_argument("--fusion", default="CAN", type=str, help="{CAN|BAN|Nan}")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Running mode: 'train' trains the model and caches features; "
             "'test' loads a saved model and re-encodes test features only.",)
    parser.add_argument(
        "--use_global_head",
        action="store_true",
        help="Include a global attention head during training.",
    )
    parser.add_argument(
        "--lambda_attn", type=float, default=0.3, help="Weight for attention loss."
    )
    parser.add_argument(
        "--attention_guided",
        type=int,
        default=1,
        help="Enable attention supervision if available.",
    )
    parser.add_argument(
        "--selected_types",
        type=int,
        nargs="*",
        default=None,
        help="Interaction type indices to include (0–6).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="S0",
        choices=["S0", "S1", "25", "28", "31", "33", "006", "043", "062", "088"],
        help=(
            "Dataset split identifier. "
            "'S0' corresponds to a random split; "
            "'25', '28', '31', and '33' denote splits based on protein sequence similarity thresholds (25%, 28%, 31%, 33%)"
            "'006', '043', '062', and '088' denote splits based on ligand sequence similarity thresholds (6%, 43%, 62%, 88%)."
        ),
    )
    parser.add_argument(
        "--attn_norm",
        type=str,
        default="row",
        choices=["row", "global", "sinkhorn"],
        help="Normalisation for attention weights in CAN layer.",
    )
    parser.add_argument(
        "--softmax_tau",
        type=float,
        default=1.0,
        help="Temperature for softmax in attention weight computation.",
    )
    parser.add_argument(
        "--sinkhorn_iters",
        type=int,
        default=5,
        help="Number of Sinkhorn iterations when attn_norm='sinkhorn'.",
    )
    parser.add_argument("--data_path", type=str, default="data/", help="Dataset root.")
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default="save_best_model/",
        help="Directory prefix for model checkpoints.",
    )
    parser.add_argument(
        "--save_name", default="fine_tune", type=str, help="Base name for saved files."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="InteractBind",
        help="Dataset name: e.g., 'InteractBind', 'BindingDB', 'Human', 'Biosnap'.",
    )

    # Avoid HF tokenisers forking too many workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return parser.parse_args()

