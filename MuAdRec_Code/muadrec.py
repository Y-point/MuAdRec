import os
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys):
        Q = self.linear_q(queries)
        K = self.linear_k(keys)
        V = self.linear_v(keys)
        
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)
        
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5
        
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)
        
        diag_vals = torch.ones_like(matmul_output[0, :, :])
        tril = torch.tril(diag_vals)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)
        
        matmul_output_sm = self.softmax(matmul_output_m2)
        
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])
        matmul_output_qm = matmul_output_sm * query_mask
        
        matmul_output_dropout = self.dropout(matmul_output_qm)
        
        output_ws = torch.bmm(matmul_output_dropout, V_)
        
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)
        
        output_res = output + queries
        
        return output_res



class SASRec_backbone(nn.Module):
    def __init__(self, device, **key_words):
        super(SASRec_backbone, self).__init__()
        
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))  
        self.seq_len = data_statis['seq_size'][0]  
        self.item_num = data_statis['item_num'][0]
        
        self.dropout = key_words["dropout_rate"]
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        self.hidden_dim = key_words["hidden_dim"]

        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_len,
            embedding_dim=self.hidden_dim
        )
        self.emb_dropout = nn.Dropout(self.dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_dim)
        self.ln_2 = nn.LayerNorm(self.hidden_dim)
        self.ln_3 = nn.LayerNorm(self.hidden_dim)
        self.mh_attn = MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout)
        self.feed_forward = PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)

    def embed_ID(self, x):
        pass
    
    def return_item_emb(self):
        pass
    
    def forward(self, sequences):
        inputs_emb = self.embed_ID(sequences)
        inputs_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        logits = ff_out[:,-1].squeeze()
        return logits
    
    def predict(self, sequences):
        state_hidden = self.forward(sequences)
        item_embs = self.return_item_emb() 
        scores = torch.matmul(state_hidden, item_embs[:-1].transpose(0, 1))
        return scores

    def calculate_ce_loss(self, sequences, target):
        seq_output = self.forward(sequences)
        item_embs = self.return_item_emb()
        logits = torch.matmul(seq_output, item_embs[:-1].transpose(0, 1))
        loss = self.ce_loss(logits, target)
        return loss
    
    def calculate_bce_loss(self, sequences, target, neg_ratio, emb_type="both"):
        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        target_neg = neg_samples.to(target.device)

        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)
        log_feats = self.forward(sequences)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats.unsqueeze(1) * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)
        loss = self.bce_loss(pos_logits, pos_labels)
        loss += self.bce_loss(neg_logits, neg_labels)

        return loss
    
    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature, emb_type="both"):
        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        target_neg = neg_samples.to(target.device)

        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)
        log_feats = self.forward(sequences)
        
        log_feats = F.normalize(log_feats, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)
        neg_logits = torch.bmm(neg_embs, log_feats.unsqueeze(-1)).squeeze(-1)
        
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits /= temperature
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss



def _load_modal_embeddings(data_dir: str, modal_name: str, model_type: str, scale: float) -> np.ndarray:
    candidates = [
        os.path.join(data_dir, f"{model_type}_{modal_name}_emb.pickle"),
        os.path.join(data_dir, f"{modal_name}_emb.pickle"),
        os.path.join(data_dir, f"{modal_name}_emb.npy"),
        os.path.join(data_dir, f"{modal_name}_emb.pt"),
    ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        if p.endswith(".pickle"):
            arr = pd.read_pickle(p)
            arr = np.stack(arr) if not isinstance(arr, np.ndarray) else arr
            return arr.astype(np.float32) * scale
        if p.endswith(".npy"):
            return np.load(p).astype(np.float32) * scale
        if p.endswith(".pt"):
            tensor = torch.load(p, map_location="cpu")
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy().astype(np.float32) * scale
    raise FileNotFoundError(f"Cannot find {modal_name} embeddings under: {data_dir}")


def _decompose_and_clip(
    embs: np.ndarray,
    hidden_dim: int,
    null_dim: Optional[int],
    null_thres: Optional[float],
    standardization: bool,
) -> Tuple[np.ndarray, int]:
    mean = np.mean(embs, axis=0)
    cov = np.cov(embs - mean, rowvar=False)
    u, s, _ = np.linalg.svd(cov, full_matrices=False)

    if null_thres is not None:
        idx_null = np.where(s <= null_thres)[0]
        nullity = len(idx_null)
    else:
        nullity = int(null_dim) if null_dim is not None else max(1, hidden_dim // 2)

    nullity = max(1, min(nullity, hidden_dim))
    clip_dim = min(hidden_dim, u.shape[1])
    p = u[:, :clip_dim]
    if standardization:
        diag = np.sqrt(1.0 / (s[:clip_dim] + 1e-12))
        p = p.dot(np.diag(diag))

    clipped = (embs - mean).dot(p)
    if clipped.shape[1] < hidden_dim:
        pad = np.zeros((clipped.shape[0], hidden_dim - clipped.shape[1]), dtype=np.float32)
        clipped = np.concatenate([clipped, pad], axis=1)
    elif clipped.shape[1] > hidden_dim:
        clipped = clipped[:, :hidden_dim]
    return clipped.astype(np.float32), nullity


@dataclass(frozen=True)
class AdaptiveNullConfig:
    enabled: bool = False
    min_dim: int = 0
    gamma: float = 1.0
    strategy: str = "inv_pop"


@dataclass(frozen=True)
class HierInjectConfig:
    enabled: bool = False
    pre_attn: bool = False
    post_attn: bool = False
    use_inject_gate: bool = False


@dataclass(frozen=True)
class MMCompleteConfig:
    use_shared_null_inject: bool = False
    use_cross_modal_null_align: bool = False
    modal_fuse_alpha: float = 0.5
    align_temperature: float = 0.1


def _load_items_pop(language_embs_path: str, item_num: int) -> Optional[np.ndarray]:
    pop_path = os.path.join(language_embs_path, "items_pop.npy")
    if not os.path.exists(pop_path):
        return None
    pop = np.load(pop_path)
    pop = pop.reshape(-1)
    if pop.shape[0] != item_num:
        return None
    return pop


def _build_effective_null_dims(
    *,
    item_num: int,
    max_null_dim: int,
    adaptive_cfg: AdaptiveNullConfig,
    language_embs_path: str,
) -> torch.Tensor:
    k = np.full((item_num,), fill_value=max_null_dim, dtype=np.int64)

    if not adaptive_cfg.enabled:
        k_pad = np.array([max_null_dim], dtype=np.int64)
        return torch.from_numpy(np.concatenate([k, k_pad], axis=0))

    pop = _load_items_pop(language_embs_path, item_num=item_num)
    if pop is None:
        k_pad = np.array([max_null_dim], dtype=np.int64)
        return torch.from_numpy(np.concatenate([k, k_pad], axis=0))

    pop = pop.astype(np.float32)
    pop = np.maximum(pop, 0.0)
    if pop.max() == pop.min():
        norm = np.zeros_like(pop)
    else:
        norm = (pop - pop.min()) / (pop.max() - pop.min() + 1e-12)

    min_dim = int(np.clip(adaptive_cfg.min_dim, 0, max_null_dim))
    inv = (1.0 - norm) ** float(adaptive_cfg.gamma)
    k = min_dim + np.round(inv * (max_null_dim - min_dim)).astype(np.int64)
    k = np.clip(k, min_dim, max_null_dim)

    k_pad = np.array([max_null_dim], dtype=np.int64)
    return torch.from_numpy(np.concatenate([k, k_pad], axis=0))


class MuAdRec(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        if key_words.get("cover", False):
            raise ValueError("MuAdRec currently requires cover=False.")

        self.data_dir = str(key_words["language_embs_path"])
        self.hidden_dim = int(key_words["hidden_dim"])
        
        self.adaptive_cfg = AdaptiveNullConfig(
            enabled=bool(key_words.get("adaptive_null_dim", False)),
            min_dim=int(key_words.get("adaptive_null_min_dim", 0)),
            gamma=float(key_words.get("adaptive_null_gamma", 1.0)),
            strategy=str(key_words.get("adaptive_null_strategy", "inv_pop")),
        )
        self.hier_cfg = HierInjectConfig(
            enabled=bool(key_words.get("hier_inject", False)),
            pre_attn=bool(key_words.get("hier_inject_pre_attn", False)),
            post_attn=bool(key_words.get("hier_inject_post_attn", False)),
            use_inject_gate=bool(key_words.get("hier_inject_use_gate", False)),
        )
        self.modal_cfg = MMCompleteConfig(
            use_shared_null_inject=bool(key_words.get("use_shared_null_inject", False)),
            use_cross_modal_null_align=bool(key_words.get("use_cross_modal_null_align", False)),
            modal_fuse_alpha=float(key_words.get("modal_fuse_alpha", 0.5)),
            align_temperature=float(key_words.get("align_temperature", 0.1)),
        )

        text_raw = _load_modal_embeddings(
            self.data_dir,
            modal_name="text",
            model_type=str(key_words.get("language_model_type", "3small")),
            scale=float(key_words.get("language_embs_scale", 40)),
        )
        image_raw = _load_modal_embeddings(
            self.data_dir,
            modal_name="image",
            model_type=str(key_words.get("language_model_type", "3small")),
            scale=float(key_words.get("image_embs_scale", key_words.get("language_embs_scale", 40))),
        )
        if text_raw.shape[0] != image_raw.shape[0]:
            raise ValueError("Text/Image embedding count mismatch.")

        self.item_num = text_raw.shape[0]
        text_clip, self.text_nullity = _decompose_and_clip(
            text_raw,
            hidden_dim=self.hidden_dim,
            null_dim=key_words.get("null_dim", 64),
            null_thres=key_words.get("null_thres", None),
            standardization=bool(key_words.get("standardization", True)),
        )
        image_clip, self.image_nullity = _decompose_and_clip(
            image_raw,
            hidden_dim=self.hidden_dim,
            null_dim=key_words.get("null_dim", 64),
            null_thres=key_words.get("null_thres", None),
            standardization=bool(key_words.get("standardization", True)),
        )

        text_pad = np.zeros((1, self.hidden_dim), dtype=np.float32)
        image_pad = np.zeros((1, self.hidden_dim), dtype=np.float32)
        text_clip = np.concatenate([text_clip, text_pad], axis=0)
        image_clip = np.concatenate([image_clip, image_pad], axis=0)

        self.text_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(text_clip, dtype=torch.float32), freeze=True, padding_idx=self.item_num)
        self.image_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(image_clip, dtype=torch.float32), freeze=True, padding_idx=self.item_num)

        shared_null_dim = min(self.text_nullity, self.image_nullity)
        self.shared_null_dim = shared_null_dim

        self._effective_text_null_dims = _build_effective_null_dims(
            item_num=int(self.item_num),
            max_null_dim=self.text_nullity,
            adaptive_cfg=self.adaptive_cfg,
            language_embs_path=str(key_words["language_embs_path"]),
        )
        self._effective_image_null_dims = _build_effective_null_dims(
            item_num=int(self.item_num),
            max_null_dim=self.image_nullity,
            adaptive_cfg=self.adaptive_cfg,
            language_embs_path=str(key_words["language_embs_path"]),
        )

        self.shared_id_embeddings = nn.Embedding(self.item_num + 1, shared_null_dim)
        init_type = str(key_words.get("ID_embs_init_type", "normal"))
        if init_type == "zeros":
            nn.init.zeros_(self.shared_id_embeddings.weight)
        else:
            nn.init.normal_(self.shared_id_embeddings.weight, 0, 1)

        self.to_text_null = nn.Linear(shared_null_dim, self.text_nullity, bias=False)
        self.to_image_null = nn.Linear(shared_null_dim, self.image_nullity, bias=False)
        
        if self.hier_cfg.use_inject_gate:
            self.pre_attn_gate = nn.Parameter(torch.tensor(1.0, device=self.device, dtype=torch.float32))
            self.post_attn_gate = nn.Parameter(torch.tensor(1.0, device=self.device, dtype=torch.float32))
        else:
            self.pre_attn_gate = None
            self.post_attn_gate = None

    def _masked_text_null(self, item_ids: torch.Tensor) -> torch.Tensor:
        shared = self.shared_id_embeddings(item_ids)
        text_null = self.to_text_null(shared)
        
        if not self.adaptive_cfg.enabled:
            return text_null
        
        eff = self._effective_text_null_dims.to(item_ids.device)
        k = eff[item_ids]
        dims = torch.arange(self.text_nullity, device=item_ids.device).view(*([1] * k.dim()), -1)
        mask = (dims < k.unsqueeze(-1)).to(text_null.dtype)
        return text_null * mask

    def _masked_image_null(self, item_ids: torch.Tensor) -> torch.Tensor:
        shared = self.shared_id_embeddings(item_ids)
        image_null = self.to_image_null(shared)
        
        if not self.adaptive_cfg.enabled:
            return image_null
        
        eff = self._effective_image_null_dims.to(item_ids.device)
        k = eff[item_ids]
        dims = torch.arange(self.image_nullity, device=item_ids.device).view(*([1] * k.dim()), -1)
        mask = (dims < k.unsqueeze(-1)).to(image_null.dtype)
        return image_null * mask

    def _inject_modal_null(self, item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        text_sem = self.text_embeddings(item_ids)
        image_sem = self.image_embeddings(item_ids)

        if not self.modal_cfg.use_shared_null_inject:
            return text_sem, image_sem

        text_null = self._masked_text_null(item_ids)
        image_null = self._masked_image_null(item_ids)

        text_fuse = text_sem.clone()
        image_fuse = image_sem.clone()
        text_fuse[..., -self.text_nullity:] = text_sem[..., -self.text_nullity:] + text_null
        image_fuse[..., -self.image_nullity:] = image_sem[..., -self.image_nullity:] + image_null
        return text_fuse, image_fuse

    def embed_ID(self, x: torch.Tensor) -> torch.Tensor:
        text_fuse, image_fuse = self._inject_modal_null(x)
        alpha = self.modal_cfg.modal_fuse_alpha
        return alpha * text_fuse + (1.0 - alpha) * image_fuse

    def return_item_emb(self) -> torch.Tensor:
        ids = torch.arange(self.item_num + 1, device=self.device, dtype=torch.long)
        return self.embed_ID(ids)

    def cross_modal_null_align_loss(self, item_ids: torch.Tensor) -> torch.Tensor:
        if not self.modal_cfg.use_cross_modal_null_align:
            return torch.tensor(0.0, device=item_ids.device)
        shared = self.shared_id_embeddings(item_ids)
        t = F.normalize(self.to_text_null(shared), p=2, dim=-1)
        v = F.normalize(self.to_image_null(shared), p=2, dim=-1)
        logits = torch.matmul(t, v.transpose(0, 1)) / self.modal_cfg.align_temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)) * 0.5

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        inputs_emb = self.embed_ID(sequences)
        inputs_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))
        seq = self.emb_dropout(inputs_emb)

        mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask

        if self.hier_cfg.enabled and self.hier_cfg.pre_attn:
            text_null = self._masked_text_null(sequences)
            image_null = self._masked_image_null(sequences)
            alpha = self.modal_cfg.modal_fuse_alpha
            null_inject = alpha * text_null + (1 - alpha) * image_null
            inject_dim = min(self.text_nullity, self.image_nullity)
            
            if self.hier_cfg.use_inject_gate and self.pre_attn_gate is not None:
                gate_weight = torch.sigmoid(self.pre_attn_gate)
                null_inject = null_inject * gate_weight
            
            seq_new = seq.clone()
            seq_new[..., -inject_dim:] = seq_new[..., -inject_dim:] + null_inject[..., -inject_dim:]
            seq = seq_new * mask

        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)

        if self.hier_cfg.enabled and self.hier_cfg.post_attn:
            text_null = self._masked_text_null(sequences)
            image_null = self._masked_image_null(sequences)
            alpha = self.modal_cfg.modal_fuse_alpha
            null_inject = alpha * text_null + (1 - alpha) * image_null
            inject_dim = min(self.text_nullity, self.image_nullity)
            
            if self.hier_cfg.use_inject_gate and self.post_attn_gate is not None:
                gate_weight = torch.sigmoid(self.post_attn_gate)
                null_inject = null_inject * gate_weight
            
            mh_attn_out_new = mh_attn_out.clone()
            mh_attn_out_new[..., -inject_dim:] = mh_attn_out_new[..., -inject_dim:] + null_inject[..., -inject_dim:]
            mh_attn_out = mh_attn_out_new * mask

        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        logits = ff_out[:, -1].squeeze()
        return logits
