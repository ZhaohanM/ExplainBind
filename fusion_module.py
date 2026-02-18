import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class CAN_Layer(nn.Module):
    """
    Cross-Attention with Grouped Aggregation (CAN)
    ----------------------------------------------
    A cross-attention block that:
      1) Groups tokens to control fusion scale,
      2) Applies multi-head attention across protein and ligand streams,
      3) Aggregates token embeddings via a configurable pooling strategy.
    """

    def __init__(self, hidden_dim, num_heads, args):
        super(CAN_Layer, self).__init__()
        self.agg_mode = args.agg_mode
        self.group_size = args.group_size            # Controls fusion granularity
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        # Attention normalisation controls
        self.attn_norm = args.attn_norm
        self.softmax_tau = args.softmax_tau
        self.sinkhorn_iters = args.sinkhorn_iters

        # Linear projections for protein stream
        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Linear projections for ligand stream
        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Optional attention pooling heads for sequence aggregation
        if self.agg_mode == "attention":
            self.attention_pooling_prot = nn.Linear(hidden_dim, 1)
            self.attention_pooling_drug = nn.Linear(hidden_dim, 1)

    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        """
        Compute attention weights under different normalisation schemes.

        Parameters
        ----------
        logits : torch.Tensor
            Shape [N, L1, L2, H]. Raw dot-product scores per head.
        mask_row : torch.Tensor
            Shape [N, L1]. Validity mask for the row axis (protein tokens).
        mask_col : torch.Tensor
            Shape [N, L2]. Validity mask for the column axis (ligand tokens).
        inf : float
            Large constant to effectively mask invalid positions.

        Returns
        -------
        torch.Tensor
            Attention weights of shape [N, L1, L2, H].
        """
        N, L1, L2, H = logits.shape

        # Scale as in standard dot-product attention
        scale = 1.0 / (self.head_size ** 0.5)
        logits = logits * scale

        # Broadcast validity to pairwise mask
        mrh = mask_row.view(N, L1, 1).repeat(1, 1, H)   # [N, L1, H]
        mch = mask_col.view(N, L2, 1).repeat(1, 1, H)   # [N, L2, H]
        mask_pair = torch.einsum('blh,bkh->blkh', mrh, mch)  # [N, L1, L2, H]

        # Exclude invalid pairs
        logits = torch.where(mask_pair, logits, logits - inf)

        tau = max(self.softmax_tau, 1e-6)

        if self.attn_norm == "row":
            # Row-wise normalisation (conditional over ligand axis)
            alpha = torch.softmax(logits / tau, dim=2)  # [N, L1, L2, H]
            mask_row4 = mask_row.view(N, L1, 1, 1).repeat(1, 1, L2, H)
            alpha = torch.where(mask_row4, alpha, torch.zeros_like(alpha))
            return alpha

        elif self.attn_norm == "global":
            # Global normalisation per head across all pairs
            alpha = (logits / tau).view(N, L1 * L2, H)
            alpha = torch.softmax(alpha, dim=1).view(N, L1, L2, H)
            mask_row4 = mask_row.view(N, L1, 1, 1).repeat(1, 1, L2, H)
            alpha = torch.where(mask_row4, alpha, torch.zeros_like(alpha))
            return alpha

        elif self.attn_norm == "sinkhorn":
            # Approximate doubly-stochastic normalisation over valid pairs
            A = (logits / tau).view(N, L1 * L2, H)
            A = torch.softmax(A, dim=1).view(N, L1, L2, H)

            mask_row4 = mask_row.view(N, L1, 1, 1).repeat(1, 1, L2, H)
            mask_col4 = mask_col.view(N, 1, L2, 1).repeat(1, L1, 1, H)
            valid = (mask_row4 & mask_col4).float()
            A = A * valid + 1e-12 * valid

            for _ in range(self.sinkhorn_iters):
                # Row normalisation on valid entries
                row_sum = (A * valid).sum(dim=2, keepdim=True) + 1e-12
                A = (A / row_sum) * valid
                # Column normalisation on valid entries
                col_sum = (A * valid).sum(dim=1, keepdim=True) + 1e-12
                A = (A / col_sum) * valid

            return A * valid

        else:
            raise ValueError(f"Unknown attn_norm: {self.attn_norm}")

    def apply_heads(self, x, n_heads, n_ch):
        """
        Reshape last dimension into [n_heads, head_dim] without copying.
        """
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def group_embeddings(self, x, mask, group_size):
        """
        Group tokens into non-overlapping windows and average within each group.

        Parameters
        ----------
        x : torch.Tensor
            Token embeddings, shape [N, L, D].
        mask : torch.Tensor
            Validity mask per token, shape [N, L] (bool).
        group_size : int
            Number of tokens per group.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Grouped embeddings [N, G, D] and grouped mask [N, G] (bool),
            where G = L // group_size.
        """
        N, L, D = x.shape
        groups = L // group_size
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2)
        mask_grouped = mask.view(N, groups, group_size).any(dim=2)
        return x_grouped, mask_grouped

    def forward(self, protein, drug, mask_prot, mask_drug):
        """
        Forward pass through grouped cross-attention with aggregation.

        Parameters
        ----------
        protein : torch.Tensor
            Protein token embeddings, [N, Lp, D].
        drug : torch.Tensor
            Ligand token embeddings, [N, Ll, D].
        mask_prot : torch.Tensor (bool)
            Valid protein tokens, [N, Lp].
        mask_drug : torch.Tensor (bool)
            Valid ligand tokens, [N, Ll].

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            - Pooled query embedding [N, 2D] (protein || ligand),
            - alpha_dp attention map [N, Gd, Gp, H] (drug→protein).
        """
        # 1) Group embeddings prior to multi-head attention
        protein_grouped, mask_prot_grouped = self.group_embeddings(
            protein, mask_prot, self.group_size
        )
        drug_grouped, mask_drug_grouped = self.group_embeddings(
            drug, mask_drug, self.group_size
        )

        # 2) Project to multi-head queries/keys/values
        query_prot = self.apply_heads(self.query_p(protein_grouped), self.num_heads, self.head_size)
        key_prot   = self.apply_heads(self.key_p(protein_grouped),   self.num_heads, self.head_size)
        value_prot = self.apply_heads(self.value_p(protein_grouped), self.num_heads, self.head_size)

        query_drug = self.apply_heads(self.query_d(drug_grouped), self.num_heads, self.head_size)
        key_drug   = self.apply_heads(self.key_d(drug_grouped),   self.num_heads, self.head_size)
        value_drug = self.apply_heads(self.value_d(drug_grouped), self.num_heads, self.head_size)

        # 3) Compute attention logits (pp, pd, dp, dd)
        logits_pp = torch.einsum('blhd,bkhd->blkh', query_prot, key_prot)
        logits_pd = torch.einsum('blhd,bkhd->blkh', query_prot, key_drug)
        logits_dp = torch.einsum('blhd,bkhd->blkh', query_drug, key_prot)
        logits_dd = torch.einsum('blhd,bkhd->blkh', query_drug, key_drug)

        # 4) Normalise to attention weights under the chosen scheme
        alpha_pp = self.alpha_logits(logits_pp, mask_prot_grouped, mask_prot_grouped)
        alpha_pd = self.alpha_logits(logits_pd, mask_prot_grouped, mask_drug_grouped)
        alpha_dp = self.alpha_logits(logits_dp, mask_drug_grouped, mask_prot_grouped)
        alpha_dd = self.alpha_logits(logits_dd, mask_drug_grouped, mask_drug_grouped)

        # 5) Cross-attend values and merge heads
        prot_embedding = (
            torch.einsum('blkh,bkhd->blhd', alpha_pp, value_prot).flatten(-2) +
            torch.einsum('blkh,bkhd->blhd', alpha_pd, value_drug).flatten(-2)
        ) / 2

        drug_embedding = (
            torch.einsum('blkh,bkhd->blhd', alpha_dp, value_prot).flatten(-2) +
            torch.einsum('blkh,bkhd->blhd', alpha_dd, value_drug).flatten(-2)
        ) / 2

        # 6) Aggregate token embeddings to fixed-size vectors
        if self.agg_mode == "cls":
            prot_embed = prot_embedding[:, 0]
            drug_embed = drug_embedding[:, 0]
        elif self.agg_mode == "mean_all_tok":
            prot_embed = prot_embedding.mean(1)
            drug_embed = drug_embedding.mean(1)
        elif self.agg_mode == "mean":
            prot_embed = (prot_embedding * mask_prot_grouped.unsqueeze(-1)).sum(1) / mask_prot_grouped.sum(-1).unsqueeze(-1)
            drug_embed = (drug_embedding * mask_drug_grouped.unsqueeze(-1)).sum(1) / mask_drug_grouped.sum(-1).unsqueeze(-1)
        elif self.agg_mode == "attention":
            # Learnable attention pooling over tokens
            attn_weights_prot = F.softmax(self.attention_pooling_prot(prot_embedding), dim=1)  # [N, Lg, 1]
            attn_weights_drug = F.softmax(self.attention_pooling_drug(drug_embedding), dim=1)  # [N, Lg, 1]
            prot_embed = (prot_embedding * attn_weights_prot).sum(dim=1)  # [N, D]
            drug_embed = (drug_embedding * attn_weights_drug).sum(dim=1)  # [N, D]
        else:
            raise NotImplementedError(f"Unknown agg_mode: {self.agg_mode}")

        # 7) Concatenate streams for downstream decoding
        query_embed = torch.cat([prot_embed, drug_embed], dim=1)
        return query_embed, alpha_dp


class BANLayer(nn.Module):
    """
    Bilinear Attention Network (BAN)
    --------------------------------
    A bilinear fusion module that computes multi-head attention maps
    between two input sequences and pools them into compact logits.

    Modified from:
    https://github.com/peizhenbai/DrugBAN/blob/main/ban.py
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)

        if 1 < k:
            # Sum-pooling implemented via average pool scaled by k
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            # Low-rank bilinear factorisation
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            # Full projection when many output heads are required
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        """
        Pool per-head attention into logits via bilinear interaction.
        """
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)            # [B, 1, D]
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(self, v, q, softmax=False):
        """
        Parameters
        ----------
        v : torch.Tensor
            Visual (or protein/ligand) features, [B, V, v_dim].
        q : torch.Tensor
            Query features, [B, Q, q_dim].
        softmax : bool
            If True, applies softmax over each attention map.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            - logits: [B, h_dim], pooled fusion output,
            - att_maps: [B, h_out, V, Q], attention maps.
        """
        v_num = v.size(1)
        q_num = q.size(1)

        if self.h_out <= self.c:
            v_ = self.v_net(v)         # [B, V, h_dim*k]
            q_ = self.q_net(q)         # [B, Q, h_dim*k]
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)   # [B, h_dim*k, V, 1]
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)   # [B, h_dim*k, 1, Q]
            d_ = torch.matmul(v_, q_)                         # [B, h_dim*k, V, Q]
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # [B, V, Q, h_out]
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)        # [B, h_out, V, Q]

        if softmax:
            p = F.softmax(att_maps.view(-1, self.h_out, v_num * q_num), dim=2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)

        # Aggregate over multi-head outputs
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits += self.attention_pooling(v_, q_, att_maps[:, i, :, :])

        logits = self.bn(logits)
        return logits, att_maps


class FCNet(nn.Module):
    """
    Fully Connected Non-linear Network
    ----------------------------------
    A stack of linear layers with optional dropout and activation.
    Modified from:
    https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if act != '':
                layers.append(getattr(nn, act)())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if act != '':
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MlPdecoder_CAN(nn.Module):
    """
    MLP Decoder for CAN Features
    ----------------------------
    A simple three-layer perceptron with batch normalisation that
    maps concatenated CAN embeddings to a binary interaction score.
    """

    def __init__(self, input_dim):
        super(MlPdecoder_CAN, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        self.bn2 = nn.BatchNorm1d(input_dim // 2)
        self.fc3 = nn.Linear(input_dim // 2, input_dim // 4)
        self.bn3 = nn.BatchNorm1d(input_dim // 4)
        self.output = nn.Linear(input_dim // 4, 1)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = torch.sigmoid(self.output(x))
        return x


class MLPdecoder_BAN(nn.Module):
    """
    MLP Decoder for BAN Features
    ----------------------------
    Maps BAN logits to a binary probability via an MLP with batch norm.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPdecoder_BAN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x
