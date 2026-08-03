from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


LABELS = [
    "O",
    "S-MENU_NM",
    "S-MENU_CNT",
    "S-MENU_NUM",
    "S-MENU_UNITPRICE",
    "S-MENU_PRICE",
    "S-MENU_DISCOUNT_PRICE",
    "S-SUBTOTAL",
    "S-DISCOUNT",
    "S-TAX",
    "S-SERVICE",
    "S-TOTAL",
    "S-CASH",
    "S-CHANGE",
    "S-CARD_PAYMENT",
    "S-EMONEY_PAYMENT",
    "S-MENUQTY_CNT",
    "S-MENUTYPE_CNT",
    "S-OTHER",
]
ID2LABEL = dict(enumerate(LABELS))


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "microsoft/layoutlmv3-base"
    model_revision: str = "main"
    max_length: int = 512
    graph_alpha: float = 0.50
    base_aux_weight: float = 0.35
    lm_dropout: float = 0.30
    gat_attention_dropout: float = 0.10
    graph_dropout: float = 0.10
    fusion_dropout: float = 0.10
    gat_layers: int = 2
    gat_heads: int = 4
    gat_edge_hidden: int = 64
    gat_ffn_multiplier: int = 1
    gat_residual_scale: float = 0.50
    spatial_input_scale: float = 0.10
    relation_types: int = 7
    edge_dim: int = 18
    max_graph_neighbors: int = 8
    column_center_threshold: float = 0.10
    column_overlap_threshold: float = 0.20
    knn_k: int = 2
    crf_weight: float = 0.65
    focal_gamma: float = 2.0
    graph_branch_dropout: float = 0.10


@dataclass(frozen=True)
class AppSettings:
    model_path: Path
    device: str
    ocr_languages: tuple[str, ...]
    ocr_gpu: bool
    eager_load: bool
    max_upload_mb: int
    max_image_pixels: int

    @classmethod
    def from_env(cls) -> "AppSettings":
        default_model = Path(__file__).resolve().parents[1] / "hybrid_model_best.zip"
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", str(default_model))).expanduser(),
            device=os.getenv("DEVICE", "auto").strip().lower(),
            ocr_languages=tuple(
                language.strip()
                for language in os.getenv("OCR_LANGUAGES", "en").split(",")
                if language.strip()
            ),
            ocr_gpu=_bool_env("OCR_GPU", True),
            eager_load=_bool_env("EAGER_LOAD_MODEL", False),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "12")),
            max_image_pixels=int(os.getenv("MAX_IMAGE_PIXELS", "25000000")),
        )


MODEL_CONFIG = ModelConfig()
