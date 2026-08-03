from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torchcrf import CRF
from transformers import LayoutLMv3Config, LayoutLMv3Model

from app.config import LABELS, MODEL_CONFIG


class WordModelBase(nn.Module):
    def __init__(self, backbone_config: LayoutLMv3Config | None, device: torch.device | None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = (
            LayoutLMv3Model(backbone_config)
            if backbone_config is not None
            else LayoutLMv3Model.from_pretrained(
                MODEL_CONFIG.model_id, revision=MODEL_CONFIG.model_revision
            )
        )
        self.hidden_size = self.backbone.config.hidden_size
        self.crf = CRF(len(LABELS), batch_first=True)

    @staticmethod
    def aggregate_words(token_embeddings, ranges_per_doc):
        return [
            torch.stack(
                [token_embeddings[index, start:end].mean(dim=0) for start, end in ranges]
            )
            for index, ranges in enumerate(ranges_per_doc)
        ]

    @staticmethod
    def pad_documents(doc_embeddings, max_words):
        hidden = doc_embeddings[0].shape[-1]
        padded = doc_embeddings[0].new_zeros((len(doc_embeddings), max_words, hidden))
        for index, embeddings in enumerate(doc_embeddings):
            padded[index, : len(embeddings)] = embeddings
        return padded

    def encode(self, batch):
        outputs = self.backbone(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            bbox=batch["bbox"].to(self.device),
            pixel_values=batch["pixel_values"].to(self.device),
        )
        text_length = batch["input_ids"].shape[1]
        return self.aggregate_words(
            outputs.last_hidden_state[:, :text_length], batch["word_ranges"]
        )

    def decode(self, emissions, mask):
        return self.crf.decode(emissions.float(), mask=mask)

    def branch_loss(self, emissions, labels, mask, class_weights):
        emissions_fp32 = emissions.float()
        crf_loss = -self.crf(emissions_fp32, labels, mask=mask, reduction="token_mean")
        logits, targets = emissions_fp32[mask], labels[mask]
        ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
        pt = torch.softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
        focal = (((1 - pt) ** MODEL_CONFIG.focal_gamma) * ce).mean()
        return MODEL_CONFIG.crf_weight * crf_loss + (1 - MODEL_CONFIG.crf_weight) * focal


class RelationEdgeGATBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_size)
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(MODEL_CONFIG.edge_dim),
            nn.Linear(MODEL_CONFIG.edge_dim, MODEL_CONFIG.gat_edge_hidden),
            nn.GELU(),
            nn.Linear(MODEL_CONFIG.gat_edge_hidden, MODEL_CONFIG.gat_edge_hidden),
        )
        self.conv = GATv2Conv(
            hidden_size,
            hidden_size // MODEL_CONFIG.gat_heads,
            heads=MODEL_CONFIG.gat_heads,
            concat=True,
            dropout=MODEL_CONFIG.gat_attention_dropout,
            edge_dim=MODEL_CONFIG.gat_edge_hidden,
            add_self_loops=True,
            share_weights=True,
        )
        ffn_hidden = MODEL_CONFIG.gat_ffn_multiplier * hidden_size
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(MODEL_CONFIG.graph_dropout),
            nn.Linear(ffn_hidden, hidden_size),
        )
        self.dropout = nn.Dropout(MODEL_CONFIG.graph_dropout)

    def forward(self, x, edge_index, edge_attr, capture_attention=False):
        normalized_x = self.pre_norm(x)
        encoded_edges = self.edge_encoder(edge_attr)
        attention = None
        if capture_attention:
            message, (attention_edges, alpha) = self.conv(
                normalized_x,
                edge_index,
                encoded_edges,
                return_attention_weights=True,
            )
            attention = {
                "edge_index": attention_edges.detach().cpu(),
                "scores": alpha.detach().float().mean(dim=-1).cpu(),
            }
        else:
            message = self.conv(normalized_x, edge_index, encoded_edges)
        x = x + MODEL_CONFIG.gat_residual_scale * self.dropout(message)
        output = x + MODEL_CONFIG.gat_residual_scale * self.dropout(
            self.ffn(self.ffn_norm(x))
        )
        return output, attention


class LayoutLMv3SymbolicRelationGATFusionCRF(WordModelBase):
    def __init__(
        self,
        graph_alpha: float | None = None,
        base_aux_weight: float | None = None,
        *,
        backbone_config: LayoutLMv3Config | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(backbone_config, device)
        hidden = self.hidden_size
        self.register_buffer(
            "_graph_alpha",
            torch.tensor(MODEL_CONFIG.graph_alpha if graph_alpha is None else graph_alpha),
        )
        self.base_aux_weight = MODEL_CONFIG.base_aux_weight if base_aux_weight is None else base_aux_weight
        self.spatial_proj = nn.Sequential(
            nn.Linear(4, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden)
        )
        self.graph_input_norm = nn.LayerNorm(hidden)
        self.gat = nn.ModuleList(
            [RelationEdgeGATBlock(hidden) for _ in range(MODEL_CONFIG.gat_layers)]
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(MODEL_CONFIG.graph_dropout),
            nn.Linear(hidden, hidden),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(MODEL_CONFIG.fusion_dropout),
            nn.Linear(hidden, hidden),
        )
        self.fusion_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(MODEL_CONFIG.lm_dropout)
        self.base_classifier = nn.Linear(hidden, len(LABELS))
        self.classifier = nn.Linear(hidden, len(LABELS))
        self.graph_mode = "original"
        self.last_attention = None
        self._current_base_emissions = None

    @property
    def graph_alpha(self) -> float:
        return float(self._graph_alpha.item())

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        if not trainable:
            self.backbone.eval()

    def loss(self, emissions, labels, mask, class_weights):
        hybrid_loss = self.branch_loss(emissions, labels, mask, class_weights)
        if self._current_base_emissions is None or self.base_aux_weight <= 0:
            return hybrid_loss
        base_loss = self.branch_loss(
            self._current_base_emissions, labels, mask, class_weights
        )
        return (hybrid_loss + self.base_aux_weight * base_loss) / (1 + self.base_aux_weight)

    def set_graph_mode(self, mode: str) -> None:
        if mode != "original":
            raise ValueError("ReceiptGraph Explorer exposes only the trained Hybrid graph")
        self.graph_mode = "original"

    def forward(self, batch):
        lm_docs = self.encode(batch)
        lm_words = torch.cat(lm_docs, dim=0)
        base_flat = self.base_classifier(self.dropout(lm_words))
        base_docs, offset = [], 0
        for lm_doc in lm_docs:
            base_docs.append(base_flat[offset : offset + len(lm_doc)])
            offset += len(lm_doc)
        base_padded = self.pad_documents(base_docs, batch["word_mask"].shape[1])
        self._current_base_emissions = base_padded
        self.last_attention = None
        branch_dropped = (
            self.training
            and self.graph_mode == "original"
            and bool(
                torch.rand((), device=lm_words.device)
                < MODEL_CONFIG.graph_branch_dropout
            )
        )
        if branch_dropped:
            return base_padded

        graph = batch["graphs"].to(self.device)
        edge_index, edge_attr = graph.edge_index, graph.edge_attr
        graph_seed = self.graph_input_norm(
            lm_words
            + MODEL_CONFIG.spatial_input_scale * self.spatial_proj(graph.spatial_pos)
        )
        graph_words = graph_seed
        for index, layer in enumerate(self.gat):
            graph_words, attention = layer(
                graph_words,
                edge_index,
                edge_attr,
                capture_attention=index == len(self.gat) - 1,
            )
            if attention is not None:
                self.last_attention = attention

        graph_delta = self.graph_proj(graph_words - graph_seed)
        fusion_residual = self.fusion_proj(torch.cat([lm_words, graph_delta], dim=-1))
        fused_words = self.fusion_norm(lm_words + self._graph_alpha * fusion_residual)
        flat_emissions = self.classifier(self.dropout(fused_words))

        docs, offset = [], 0
        for lm_doc in lm_docs:
            docs.append(flat_emissions[offset : offset + len(lm_doc)])
            offset += len(lm_doc)
        return self.pad_documents(docs, batch["word_mask"].shape[1])
