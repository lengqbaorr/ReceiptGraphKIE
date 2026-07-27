# ReceiptGraphKIE

Graph-enhanced key information extraction for receipt understanding on CORD-1000. The project compares a LayoutLMv3–CRF baseline with a hybrid model that combines LayoutLMv3, a two-layer symbolic Relation-GATv2, hidden-state fusion, and a word-level CRF.

## Architecture

```text
Receipt image + oracle text/bounding boxes
                    │
                LayoutLMv3
                    │
           Word-level representations
             ┌──────┴──────┐
       Base classifier   Symbolic Relation-GATv2
                              │
                         Hidden fusion
                              │
                     Fusion classifier + CRF
                              │
                       Receipt entities
```

The symbolic graph connects words using exact line membership, cross-line column continuity, directional spatial relations, and local KNN fallback edges. Four development ablations—`original`, `shuffled`, `self_loop_only`, and `graph_off`—measure whether the graph topology provides useful information.

## Results

Entity Macro F1 is computed over 18 entity classes, excluding `O`.

| Model | Seeds | Dev Macro F1 | Test Macro F1 | Test Micro F1 |
|---|---:|---:|---:|---:|
| LayoutLMv3 + Word-CRF | 3 | 0.9256 ± 0.0100 | 0.9233 ± 0.0207 | 0.9712 ± 0.0043 |
| LayoutLMv3 + Relation-GATv2 + Word-CRF | 5 | 0.9247 ± 0.0049 | **0.9398 ± 0.0161** | **0.9764 ± 0.0037** |

The hybrid model improves Test Macro F1 by **1.65 percentage points** over the baseline. On development data, the original graph outperforms shuffled edges by **2.0159 ± 0.4385 points**, while disabling the graph reduces Macro F1 by **0.9464 ± 0.3713 points**.

## Repository

```text
.
├── Baseline.ipynb   # LayoutLMv3 + word-level CRF, three seeds
├── Hybrid.ipynb     # Symbolic Relation-GATv2 hybrid, five seeds and ablations
└── CORD1000/        # CORD train/dev/test images and annotations
```

## Reproducing the Experiments

1. Upload the repository or notebooks to Kaggle.
2. Enable a GPU runtime and attach the CORD-1000 dataset.
3. Run `Baseline.ipynb` or `Hybrid.ipynb` from top to bottom.
4. Download the generated artifact ZIP from `/kaggle/working/`.

Each seed is initialized independently. Checkpoints are selected exclusively by development Entity Macro F1, and the test split is evaluated only after model selection. Results are reported as mean ± sample standard deviation.

## Evaluation Scope

The notebooks use reference CORD text and bounding boxes with OCR disabled. The reported scores therefore evaluate the KIE component, not an end-to-end OCR-to-extraction pipeline. The baseline and hybrid use different seed counts, so their aggregate comparison should not be interpreted as a paired statistical superiority test.

## References

- [CORD](https://github.com/clovaai/cord)
- [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
