from __future__ import annotations

import shutil
import tempfile
import threading
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch_geometric.data import Batch
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)

from app.config import AppSettings, ID2LABEL, MODEL_CONFIG
from app.graph import RELATION_TO_ID, build_spatial_graph, normalize_box
from app.model import LayoutLMv3SymbolicRelationGATFusionCRF
from app.ocr import EasyOCRAdapter, OCRWord


def resolve_checkpoint(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix.lower() != ".zip":
        return path

    cache_dir = Path(tempfile.gettempdir()) / "receiptgraphkie"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{path.stem}.pt"
    if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
        return target
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".pt")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one .pt file in {path}, found {len(members)}")
        with archive.open(members[0]) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    return target


def load_processor() -> LayoutLMv3Processor:
    """Build the processor without requiring the newer processor_config.json."""
    common = {"revision": MODEL_CONFIG.model_revision}
    try:
        image_processor = LayoutLMv3ImageProcessor.from_pretrained(
            MODEL_CONFIG.model_id,
            apply_ocr=False,
            local_files_only=True,
            **common,
        )
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
            MODEL_CONFIG.model_id, local_files_only=True, **common
        )
    except OSError:
        image_processor = LayoutLMv3ImageProcessor.from_pretrained(
            MODEL_CONFIG.model_id, apply_ocr=False, **common
        )
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
            MODEL_CONFIG.model_id, **common
        )
    return LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)


def build_batch(processor, image: Image.Image, words: list[dict[str, Any]]) -> tuple[dict, list[dict]]:
    width, height = image.size
    encoding = processor(
        image,
        [word["text"] for word in words],
        boxes=[normalize_box(word["box"], width, height) for word in words],
        truncation=True,
        max_length=MODEL_CONFIG.max_length,
        padding="max_length",
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    active_ids = sorted({word_id for word_id in word_ids if word_id is not None})
    active_words = [words[word_id] for word_id in active_ids]
    ranges = []
    for word_id in active_ids:
        positions = [position for position, current in enumerate(word_ids) if current == word_id]
        ranges.append((positions[0], positions[-1] + 1))
    graph = build_spatial_graph(active_words, image.size)
    batch = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bbox": encoding["bbox"],
        "pixel_values": encoding["pixel_values"],
        "word_ranges": [ranges],
        "word_mask": torch.ones((1, len(active_words)), dtype=torch.bool),
        "graphs": Batch.from_data_list([graph]),
    }
    return batch, active_words


def postprocess(words: list[dict[str, Any]], labels: list[str], confidences: list[float]) -> dict:
    tokens = []
    entities = []
    current = None
    fields: dict[str, list[str]] = defaultdict(list)
    for word, label, confidence in zip(words, labels, confidences):
        token = {
            "text": word["text"],
            "box": word["box"],
            "line_id": word["line_id"],
            "label": label,
            "confidence": round(float(confidence), 4),
            "ocr_confidence": round(float(word.get("ocr_confidence", 1.0)), 4),
        }
        tokens.append(token)
        if label == "O":
            current = None
            continue
        field = label.removeprefix("S-")
        if current and current["field"] == field and current["line_id"] == word["line_id"]:
            current["text"] += " " + word["text"]
            current["box"] = [
                min(current["box"][0], word["box"][0]),
                min(current["box"][1], word["box"][1]),
                max(current["box"][2], word["box"][2]),
                max(current["box"][3], word["box"][3]),
            ]
            current["confidence"] = round(min(current["confidence"], float(confidence)), 4)
        else:
            current = {
                "field": field,
                "text": word["text"],
                "box": word["box"].copy(),
                "line_id": word["line_id"],
                "confidence": round(float(confidence), 4),
            }
            entities.append(current)
    for entity in entities:
        fields[entity["field"]].append(entity["text"])
    grouped_fields = dict(fields)
    return {
        "fields": grouped_fields,
        "structured": build_structured_receipt(entities),
        "entities": entities,
        "tokens": tokens,
    }


def build_structured_receipt(entities: list[dict[str, Any]]) -> dict:
    by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_line: dict[str, dict[str, str]] = defaultdict(dict)
    menu_fields = {"MENU_NM", "MENU_CNT", "MENU_NUM", "MENU_UNITPRICE", "MENU_PRICE", "MENU_DISCOUNT_PRICE"}
    for entity in entities:
        by_field[entity["field"]].append(entity)
        if entity["field"] in menu_fields:
            by_line[entity["line_id"]][entity["field"]] = entity["text"]

    def first(field: str) -> str:
        return by_field[field][0]["text"] if by_field.get(field) else ""

    items = []
    for line in by_line.values():
        if not line.get("MENU_NM") and not line.get("MENU_PRICE"):
            continue
        items.append(
            {
                "name": line.get("MENU_NM", ""),
                "quantity": line.get("MENU_CNT", ""),
                "number": line.get("MENU_NUM", ""),
                "unit_price": line.get("MENU_UNITPRICE", ""),
                "price": line.get("MENU_PRICE", ""),
                "discount_price": line.get("MENU_DISCOUNT_PRICE", ""),
            }
        )
    payment_method = (
        "card" if by_field.get("CARD_PAYMENT") else
        "electronic_money" if by_field.get("EMONEY_PAYMENT") else
        "cash" if by_field.get("CASH") else ""
    )
    return {
        "merchant": {"name": "", "address": ""},
        "items": items,
        "subtotal": first("SUBTOTAL"),
        "tax": first("TAX"),
        "service": first("SERVICE"),
        "discount": first("DISCOUNT"),
        "total": first("TOTAL"),
        "payment": {
            "method": payment_method,
            "cash": first("CASH"),
            "change": first("CHANGE"),
            "card": first("CARD_PAYMENT"),
            "electronic_money": first("EMONEY_PAYMENT"),
        },
        "other": [entity["text"] for entity in by_field.get("OTHER", [])],
    }


def graph_payload(words, edge_index, edge_attr, attention=None) -> dict:
    relation_names = [name for name, _ in sorted(RELATION_TO_ID.items(), key=lambda item: item[1])]
    attention_map: dict[tuple[int, int], list[float]] = defaultdict(list)
    if attention:
        indices = attention["edge_index"].tolist()
        for source, target, score in zip(indices[0], indices[1], attention["scores"].tolist()):
            attention_map[(source, target)].append(float(score))
    edges = []
    edge_values = edge_attr.detach().cpu().tolist()
    edge_indices = edge_index.detach().cpu().tolist()
    for index, (source, target) in enumerate(zip(edge_indices[0], edge_indices[1])):
        values = edge_values[index]
        relations = [
            relation_names[relation_id]
            for relation_id, enabled in enumerate(values[11:])
            if enabled > 0.5
        ]
        scores = attention_map.get((source, target), [])
        edges.append(
            {
                "source": source,
                "target": target,
                "relations": relations,
                "attention": round(sum(scores) / len(scores), 6) if scores else None,
                "geometry": {
                    "dx": round(values[0], 5),
                    "dy": round(values[1], 5),
                    "distance": round(values[2], 5),
                    "vertical_overlap": round(values[6], 5),
                    "horizontal_overlap": round(values[7], 5),
                    "same_line": bool(values[8]),
                    "angle_sin": round(values[9], 5),
                    "angle_cos": round(values[10], 5),
                },
            }
        )
    nodes = [
        {
            "id": index,
            "text": word["text"],
            "box": word["box"],
            "line_id": word["line_id"],
        }
        for index, word in enumerate(words)
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "relation_types": relation_names,
    }


class ReceiptKIEService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.device = self._select_device(settings.device)
        self.model = None
        self.processor = None
        self.ocr = None
        self.load_error: str | None = None
        self._load_lock = threading.Lock()
        self._ocr_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda was requested but CUDA is unavailable")
        return torch.device(requested)

    @property
    def ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> None:
        if self.ready:
            return
        with self._load_lock:
            if self.ready:
                return
            try:
                checkpoint = resolve_checkpoint(self.settings.model_path)
                config_path = Path(__file__).resolve().parent / "assets" / "layoutlmv3_config.json"
                backbone_config = LayoutLMv3Config.from_json_file(config_path)
                processor = load_processor()
                model = LayoutLMv3SymbolicRelationGATFusionCRF(
                    backbone_config=backbone_config, device=self.device
                )
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
                model.to(self.device).eval()
                self.model, self.processor = model, processor
                self.load_error = None
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                raise

    @torch.inference_mode()
    def analyze_words(
        self,
        image: Image.Image,
        words: list[dict[str, Any]],
        ground_truth: list[str] | None = None,
        source: str = "upload",
    ) -> dict:
        self.load()
        started = time.perf_counter()
        image = image.convert("RGB")
        if not words:
            raise ValueError("No words or bounding boxes were detected")
        with self._inference_lock:
            batch, active_words = build_batch(self.processor, image, words)
            mask = batch["word_mask"].to(self.device)
            self.model.set_graph_mode("original")
            used_edge_index = batch["graphs"].edge_index.detach().cpu()
            used_edge_attr = batch["graphs"].edge_attr.detach().cpu()
            emissions = self.model(batch)
            path = self.model.decode(emissions, mask)[0]
            probabilities = torch.softmax(emissions[0, : len(path)].float(), dim=-1)
            confidence = probabilities[
                torch.arange(len(path), device=probabilities.device),
                torch.tensor(path, device=probabilities.device),
            ].cpu().tolist()
            result = postprocess(
                active_words, [ID2LABEL[index] for index in path], confidence
            )
            result["graph"] = graph_payload(
                active_words,
                used_edge_index,
                used_edge_attr,
                self.model.last_attention,
            )

        active_ground_truth = ground_truth[: len(active_words)] if ground_truth else None
        return {
            "source": source,
            "result": result,
            "ground_truth_available": active_ground_truth is not None,
            "image": {"width": image.width, "height": image.height},
            "input_word_count": len(words),
            "model_word_count": len(active_words),
            "truncated": len(active_words) < len(words),
            "device": str(self.device),
            "processing_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    @torch.inference_mode()
    def extract(self, image: Image.Image) -> dict:
        self.load()
        if self.ocr is None:
            with self._ocr_lock:
                if self.ocr is None:
                    self.ocr = EasyOCRAdapter(
                        self.settings.ocr_languages,
                        use_gpu=self.settings.ocr_gpu and self.device.type == "cuda",
                    )
        image = image.convert("RGB")
        ocr_words: list[OCRWord] = self.ocr.read(image)
        words = [word.as_model_input() for word in ocr_words]
        analysis = self.analyze_words(
            image,
            words,
            source="real_receipt_ocr",
        )
        result = analysis["result"]
        result.update(
            {
                "image": {"width": image.width, "height": image.height},
                "ocr_word_count": len(words),
                "model_word_count": analysis["model_word_count"],
                "truncated": analysis["truncated"],
                "device": str(self.device),
                "processing_ms": analysis["processing_ms"],
            }
        )
        return result
