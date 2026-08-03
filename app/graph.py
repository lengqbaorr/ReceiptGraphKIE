from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from app.config import MODEL_CONFIG


RELATION_TO_ID = {
    "KNN": 0,
    "LEFT": 1,
    "RIGHT": 2,
    "ABOVE": 3,
    "BELOW": 4,
    "SAME_LINE": 5,
    "NEXT_LINE_COLUMN": 6,
}


def normalize_box(box: list[int], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    values = [1000 * x0 / width, 1000 * y0 / height, 1000 * x1 / width, 1000 * y1 / height]
    return [max(0, min(1000, int(value))) for value in values]
INVERSE_RELATION = {
    "KNN": "KNN",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
    "ABOVE": "BELOW",
    "BELOW": "ABOVE",
    "SAME_LINE": "SAME_LINE",
    "NEXT_LINE_COLUMN": "NEXT_LINE_COLUMN",
}


def vertical_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return float(overlap / max(min(a[3] - a[1], b[3] - b[1]), 1.0))


def horizontal_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return float(overlap / max(min(a[2] - a[0], b[2] - b[0]), 1.0))


def horizontally_aligned(a: np.ndarray, b: np.ndarray, image_width: int) -> bool:
    center_distance = abs((a[0] + a[2] - b[0] - b[2]) / 2) / image_width
    return (
        horizontal_overlap_ratio(a, b) >= MODEL_CONFIG.column_overlap_threshold
        or center_distance <= MODEL_CONFIG.column_center_threshold
    )


def build_spatial_graph(
    words: list[dict[str, Any]], image_size: tuple[int, int], k: int | None = None
) -> Data:
    """Reproduce the graph construction used by Hybrid.ipynb exactly."""
    width, height = image_size
    boxes = np.asarray([word["box"] for word in words], dtype=np.float32)
    n = len(words)
    centers = np.column_stack(
        (
            (boxes[:, 0] + boxes[:, 2]) / (2 * width),
            (boxes[:, 1] + boxes[:, 3]) / (2 * height),
        )
    )
    widths = (boxes[:, 2] - boxes[:, 0]) / width
    heights = (boxes[:, 3] - boxes[:, 1]) / height
    edge_relations: dict[tuple[int, int], set[int]] = defaultdict(set)

    def connect(i: int, j: int, relation: str) -> None:
        if i == j:
            return
        edge_relations[(i, j)].add(RELATION_TO_ID[relation])
        edge_relations[(j, i)].add(RELATION_TO_ID[INVERSE_RELATION[relation]])

    line_to_nodes: dict[str, list[int]] = defaultdict(list)
    for index, word in enumerate(words):
        line_to_nodes[str(word["line_id"])].append(index)
    ordered_lines = sorted(
        line_to_nodes.values(), key=lambda nodes: float(np.mean(centers[nodes, 1]))
    )

    for nodes in ordered_lines:
        ordered_nodes = sorted(nodes, key=lambda index: centers[index, 0])
        for left_node, right_node in zip(ordered_nodes[:-1], ordered_nodes[1:]):
            connect(left_node, right_node, "SAME_LINE")
            connect(left_node, right_node, "RIGHT")

    for upper_nodes, lower_nodes in zip(ordered_lines[:-1], ordered_lines[1:]):
        for i in upper_nodes:
            candidates = [
                j for j in lower_nodes if horizontally_aligned(boxes[i], boxes[j], width)
            ]
            if candidates:
                j = min(candidates, key=lambda node: abs(centers[node, 0] - centers[i, 0]))
                connect(i, j, "NEXT_LINE_COLUMN")
                connect(i, j, "BELOW")

    if n > 1:
        for i in range(n):
            left = [
                j
                for j in range(n)
                if centers[j, 0] < centers[i, 0]
                and words[j]["line_id"] == words[i]["line_id"]
            ]
            right = [
                j
                for j in range(n)
                if centers[j, 0] > centers[i, 0]
                and words[j]["line_id"] == words[i]["line_id"]
            ]
            if left:
                connect(i, min(left, key=lambda j: centers[i, 0] - centers[j, 0]), "LEFT")
            if right:
                connect(i, min(right, key=lambda j: centers[j, 0] - centers[i, 0]), "RIGHT")

        for i in range(n):
            current = {dst for src, dst in edge_relations if src == i}
            free_slots = max(MODEL_CONFIG.max_graph_neighbors - len(current), 0)
            distances = np.linalg.norm(centers - centers[i], axis=1)
            candidates = [
                int(j)
                for j in np.argsort(distances)
                if j != i and int(j) not in current
            ]
            for j in candidates[: min(k if k is not None else MODEL_CONFIG.knn_k, free_slots)]:
                connect(i, j, "KNN")

    symbolic_ids = {RELATION_TO_ID["SAME_LINE"], RELATION_TO_ID["NEXT_LINE_COLUMN"]}
    directional_ids = {
        RELATION_TO_ID["LEFT"],
        RELATION_TO_ID["RIGHT"],
        RELATION_TO_ID["ABOVE"],
        RELATION_TO_ID["BELOW"],
    }
    bounded_relations: dict[tuple[int, int], set[int]] = {}
    for i in range(n):
        destinations = [j for src, j in edge_relations if src == i]

        def priority(j: int) -> tuple[int, float]:
            relations = edge_relations[(i, j)]
            relation_priority = 0 if relations & symbolic_ids else 1 if relations & directional_ids else 2
            return relation_priority, float(np.linalg.norm(centers[j] - centers[i]))

        destinations.sort(key=priority)
        for j in destinations[: MODEL_CONFIG.max_graph_neighbors]:
            bounded_relations[(i, j)] = edge_relations[(i, j)]

    src: list[int] = []
    dst: list[int] = []
    features: list[list[float]] = []
    for (i, j), relation_ids in sorted(bounded_relations.items()):
        dx = float(centers[j, 0] - centers[i, 0])
        dy = float(centers[j, 1] - centers[i, 1])
        angle = math.atan2(dy, dx)
        relation_one_hot = [
            float(relation_id in relation_ids)
            for relation_id in range(MODEL_CONFIG.relation_types)
        ]
        features.append(
            [
                dx,
                dy,
                float(np.hypot(dx, dy)),
                float(np.clip(np.log(max(widths[j], 1e-6) / max(widths[i], 1e-6)), -4, 4)),
                float(np.clip(np.log(max(heights[j], 1e-6) / max(heights[i], 1e-6)), -4, 4)),
                float(
                    np.clip(
                        np.log(
                            max(widths[j] * heights[j], 1e-6)
                            / max(widths[i] * heights[i], 1e-6)
                        ),
                        -4,
                        4,
                    )
                ),
                vertical_overlap_ratio(boxes[i], boxes[j]),
                horizontal_overlap_ratio(boxes[i], boxes[j]),
                float(words[i]["line_id"] == words[j]["line_id"]),
                math.sin(angle),
                math.cos(angle),
                *relation_one_hot,
            ]
        )
        src.append(i)
        dst.append(j)

    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(features, dtype=torch.float32)
        if features
        else torch.empty((0, MODEL_CONFIG.edge_dim), dtype=torch.float32)
    )
    spatial = torch.tensor(
        np.column_stack((centers, widths, heights)), dtype=torch.float32
    )
    return Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        spatial_pos=spatial,
        num_nodes=n,
    )
