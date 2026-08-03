from app.inference import postprocess


def test_postprocess_merges_same_field_on_same_line():
    words = [
        {"text": "ICE", "box": [0, 0, 20, 10], "line_id": "0"},
        {"text": "TEA", "box": [22, 0, 45, 10], "line_id": "0"},
        {"text": "5.00", "box": [60, 0, 90, 10], "line_id": "0"},
    ]
    result = postprocess(
        words,
        ["S-MENU_NM", "S-MENU_NM", "S-MENU_PRICE"],
        [0.9, 0.8, 0.95],
    )
    assert result["fields"]["MENU_NM"] == ["ICE TEA"]
    assert result["fields"]["MENU_PRICE"] == ["5.00"]
    assert result["entities"][0]["confidence"] == 0.8

