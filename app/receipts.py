from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


CORD_CATEGORY_TO_LABEL = {
    "menu.nm": "S-MENU_NM", "menu.sub_nm": "S-MENU_NM",
    "menu.cnt": "S-MENU_CNT", "menu.sub_cnt": "S-MENU_CNT",
    "menu.num": "S-MENU_NUM", "menu.unitprice": "S-MENU_UNITPRICE",
    "menu.price": "S-MENU_PRICE", "menu.sub_price": "S-MENU_PRICE",
    "menu.discountprice": "S-MENU_DISCOUNT_PRICE",
    "sub_total.subtotal_price": "S-SUBTOTAL",
    "sub_total.discount_price": "S-DISCOUNT",
    "sub_total.tax_price": "S-TAX", "sub_total.service_price": "S-SERVICE",
    "total.total_price": "S-TOTAL", "total.cashprice": "S-CASH",
    "total.changeprice": "S-CHANGE", "total.creditcardprice": "S-CARD_PAYMENT",
    "total.emoneyprice": "S-EMONEY_PAYMENT", "total.menuqty_cnt": "S-MENUQTY_CNT",
    "total.menutype_cnt": "S-MENUTYPE_CNT",
    "sub_total.etc": "S-OTHER", "total.total_etc": "S-OTHER",
    "menu.etc": "S-OTHER", "menu.sub_etc": "S-OTHER",
    "menu.itemsubtotal": "O", "menu.vatyn": "O",
    "sub_total.othersvc_price": "O", "void_menu.nm": "O",
    "void_menu.price": "O", "menu.sub_unitprice": "O",
}


def find_cord_root(search_root: Path) -> Path | None:
    direct = search_root / "CORD1000" / "CORD" / "CORD"
    bundled = search_root / "app" / "assets" / "cord_samples"
    def valid(candidate: Path) -> bool:
        return all(
            (candidate / split / "json").is_dir()
            and (candidate / split / "image").is_dir()
            for split in ("train", "dev", "test")
        )

    if valid(direct):
        return direct.resolve()
    if (bundled / "dev" / "json").is_dir() and (bundled / "dev" / "image").is_dir():
        return bundled.resolve()
    for path in search_root.rglob("train/json"):
        candidate = path.parent.parent
        if valid(candidate):
            return candidate.resolve()
    return None


def quad_to_bbox(quad: dict, width: int, height: int) -> list[int]:
    xs = [quad[f"x{i}"] for i in range(1, 5)]
    ys = [quad[f"y{i}"] for i in range(1, 5)]
    x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
    x1, y1 = min(width, int(max(xs))), min(height, int(max(ys)))
    return [min(x0, width - 1), min(y0, height - 1), max(x0 + 1, x1), max(y0 + 1, y1)]


class CORDRepository:
    def __init__(self, root: Path | None):
        self.root = root

    @property
    def available(self) -> bool:
        return self.root is not None and self.root.is_dir()

    def list_samples(self, split: str = "dev", limit: int = 30) -> list[dict]:
        self._validate_split(split)
        if not self.available:
            return []
        samples = []
        for json_path in sorted((self.root / split / "json").glob("*.json"))[:limit]:
            image_path = self.root / split / "image" / f"{json_path.stem}.png"
            if image_path.exists():
                samples.append({"id": json_path.stem, "split": split, "filename": image_path.name})
        return samples

    def load(self, split: str, receipt_id: str) -> tuple[Image.Image, list[dict], list[str], Path]:
        self._validate_split(split)
        if not self.available:
            raise FileNotFoundError("CORD dataset is not available")
        safe_id = Path(receipt_id).name
        if safe_id != receipt_id:
            raise ValueError("Invalid receipt ID")
        json_path = self.root / split / "json" / f"{safe_id}.json"
        image_path = self.root / split / "image" / f"{safe_id}.png"
        if not json_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"CORD sample not found: {split}/{safe_id}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        width = int(payload["meta"]["image_size"]["width"])
        height = int(payload["meta"]["image_size"]["height"])
        words = []
        for line_index, line in enumerate(payload.get("valid_line", [])):
            label = CORD_CATEGORY_TO_LABEL.get(line.get("category", ""), "O")
            for word in line.get("words", []):
                text = word.get("text", "").strip()
                if not text:
                    continue
                raw_row_id = word.get("row_id")
                words.append(
                    {
                        "text": text,
                        "box": quad_to_bbox(word["quad"], width, height),
                        "line_id": f"row_{raw_row_id}" if raw_row_id is not None else f"valid_line_{line_index}",
                        "ocr_confidence": 1.0,
                        "ground_truth": label,
                    }
                )
        words.sort(key=lambda item: (item["box"][1], item["box"][0]))
        labels = [word["ground_truth"] for word in words]
        return Image.open(image_path).convert("RGB"), words, labels, image_path

    @staticmethod
    def _validate_split(split: str) -> None:
        if split not in {"train", "dev", "test"}:
            raise ValueError("Split must be train, dev or test")
