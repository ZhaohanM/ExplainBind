import logging
import sys

sys.path.append("../")

import torch
import torch.nn as nn
from fusion_module import CAN_Layer, MlPdecoder_CAN

LOGGER = logging.getLogger(__name__)


# =============================================================================
# ExplainPLI
# =============================================================================
class ExplainPLI(nn.Module):

    def __init__(self, prot_out_dim: int, drug_out_dim: int, args):
        super().__init__()

        self.fusion = args.fusion

        # Project encoder outputs to shared hidden space
        self.prot_reg = nn.Linear(prot_out_dim, 768)
        self.drug_reg = nn.Linear(drug_out_dim, 768)

        if self.fusion == "CAN":
            self.can_layer = CAN_Layer(hidden_dim=768, num_heads=8, args=args)
            self.mlp_classifier = MlPdecoder_CAN(input_dim=1536)

        elif self.fusion == "Nan":
            self.mlp_classifier_nan = MlPdecoder_CAN(
                input_dim=prot_out_dim + drug_out_dim
            )
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion!r}")
    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(
        self,
        prot_embed: torch.Tensor,
        prot_mask: torch.Tensor,
        drug_embed: torch.Tensor,
        drug_mask: torch.Tensor,
    ):
        att = None

        if self.fusion == "Nan":
            prot_vec = prot_embed.mean(dim=1)
            drug_vec = drug_embed.mean(dim=1)
            joint_embed = torch.cat([prot_vec, drug_vec], dim=1)
            score = self.mlp_classifier_nan(joint_embed)

        else:
            prot_fused = self.prot_reg(prot_embed)
            drug_fused = self.drug_reg(drug_embed)

            if self.fusion == "CAN":
                joint_embed, att = self.can_layer(
                    prot_fused, drug_fused, prot_mask, drug_mask
                )
            elif self.fusion == "BAN":
                joint_embed, att = self.ban_layer(prot_fused, drug_fused)
            else:
                raise ValueError(f"Unknown fusion mode: {self.fusion!r}")

            score = self.mlp_classifier(joint_embed)

        return score, att, joint_embed


class EncoderWrapper(nn.Module):
    """
    token ids -> encoder embeddings -> ExplainPLI

    Encoders are NOT frozen: full finetuning enabled.
    """

    def __init__(
        self,
        prot_encoder: nn.Module,
        drug_encoder: nn.Module,
        pli_model: nn.Module,
    ):
        super().__init__()

        self.prot_encoder = prot_encoder
        self.drug_encoder = drug_encoder
        self.pli_model = pli_model

    def forward(
        self,
        prot_ids: torch.Tensor,
        prot_mask: torch.Tensor,
        drug_ids: torch.Tensor,
        drug_mask: torch.Tensor,
    ):
        # ---------------------------------------------------------------------
        # Encode (WITH gradient)
        # ---------------------------------------------------------------------
        prot_out = self.prot_encoder(
            input_ids=prot_ids,
            attention_mask=prot_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        prot_embed = prot_out.hidden_states[-1]

        drug_out = self.drug_encoder(
            input_ids=drug_ids,
            attention_mask=drug_mask,
            return_dict=True,
        )
        drug_embed = drug_out.last_hidden_state

        # ---------------------------------------------------------------------
        # Fusion + prediction
        # ---------------------------------------------------------------------
        return self.pli_model(
            prot_embed,
            prot_mask,
            drug_embed,
            drug_mask,
        )
