from app.config import MODEL_CONFIG
from app.graph import build_spatial_graph


def test_graph_shapes_for_receipt_line():
    words = [
        {"text": "Coffee", "box": [10, 10, 80, 30], "line_id": "line_0"},
        {"text": "2", "box": [100, 10, 110, 30], "line_id": "line_0"},
        {"text": "10.00", "box": [150, 10, 200, 30], "line_id": "line_0"},
    ]
    graph = build_spatial_graph(words, (220, 100))
    assert graph.num_nodes == 3
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == MODEL_CONFIG.edge_dim
    assert graph.spatial_pos.shape == (3, 4)


def test_single_word_graph_has_empty_edges():
    graph = build_spatial_graph(
        [{"text": "TOTAL", "box": [10, 10, 60, 30], "line_id": "line_0"}],
        (100, 100),
    )
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, MODEL_CONFIG.edge_dim)

