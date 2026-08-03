from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class OCRWord:
    text: str
    box: tuple[int, int, int, int]
    line_id: str
    confidence: float

    def as_model_input(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "box": list(self.box),
            "line_id": self.line_id,
            "ocr_confidence": self.confidence,
        }


def _vertical_overlap(a: list[int], b: list[int]) -> float:
    overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return overlap / max(min(a[3] - a[1], b[3] - b[1]), 1)


def assign_reading_order(items: list[dict[str, Any]]) -> list[OCRWord]:
    """Sort OCR boxes and infer stable line IDs required by the symbolic graph."""
    items = sorted(items, key=lambda item: ((item["box"][1] + item["box"][3]) / 2, item["box"][0]))
    lines: list[dict[str, Any]] = []
    for item in items:
        box = item["box"]
        best_index = None
        best_overlap = 0.0
        for index, line in enumerate(lines):
            overlap = _vertical_overlap(box, line["box"])
            if overlap >= 0.45 and overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index is None:
            lines.append({"box": box.copy(), "items": [item]})
        else:
            line = lines[best_index]
            line["items"].append(item)
            line["box"] = [
                min(line["box"][0], box[0]),
                min(line["box"][1], box[1]),
                max(line["box"][2], box[2]),
                max(line["box"][3], box[3]),
            ]

    lines.sort(key=lambda line: (line["box"][1], line["box"][0]))
    words: list[OCRWord] = []
    for line_index, line in enumerate(lines):
        for item in sorted(line["items"], key=lambda current: current["box"][0]):
            words.append(
                OCRWord(
                    text=item["text"],
                    box=tuple(item["box"]),
                    line_id=f"ocr_line_{line_index}",
                    confidence=float(item["confidence"]),
                )
            )
    return words


class EasyOCRAdapter:
    def __init__(self, languages: tuple[str, ...], use_gpu: bool):
        import easyocr

        gpu = use_gpu and torch.cuda.is_available()
        self.reader = easyocr.Reader(list(languages), gpu=gpu)

    def read(self, image: Image.Image) -> list[OCRWord]:
        raw = self.reader.readtext(np.asarray(image), detail=1, paragraph=False)
        items = []
        width, height = image.size
        for quad, text, confidence in raw:
            cleaned = str(text).strip()
            if not cleaned:
                continue
            xs = [float(point[0]) for point in quad]
            ys = [float(point[1]) for point in quad]
            x0 = max(0, min(int(min(xs)), width - 1))
            y0 = max(0, min(int(min(ys)), height - 1))
            x1 = max(x0 + 1, min(int(max(xs)), width))
            y1 = max(y0 + 1, min(int(max(ys)), height))
            items.append(
                {
                    "text": cleaned,
                    "box": [x0, y0, x1, y1],
                    "confidence": float(confidence),
                }
            )
        return assign_reading_order(items)

