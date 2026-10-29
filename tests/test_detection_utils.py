from types import SimpleNamespace

from utils.detection_utils import get_zone, process_box, refine_ungulate_label

CLASS_NAMES = {0: "person", 1: "zebra", 2: "horse"}
ROI = (100, 50, 400, 250)  # x1, y1, x2, y2 in frame pixels
FRAME_W, FRAME_H = 500, 300


def make_box(cls_id, conf, xyxy, track_id=None):
    return SimpleNamespace(
        cls=[cls_id], conf=[conf], xyxy=[xyxy], id=None if track_id is None else [track_id]
    )


def test_get_zone_splits_span_into_thirds():
    assert get_zone(100, 100, 400) == "left"
    assert get_zone(199, 100, 400) == "left"
    assert get_zone(200, 100, 400) == "center"
    assert get_zone(300, 100, 400) == "right"


def test_refine_ungulate_label():
    assert refine_ungulate_label(0, 0, 10, 20, FRAME_W, FRAME_H) == "antelope_like"
    assert refine_ungulate_label(0, 0, 100, 50, FRAME_W, FRAME_H) == "large_ungulate"
    assert refine_ungulate_label(0, 0, 20, 10, FRAME_W, FRAME_H) == "ungulate"


def test_process_box_offsets_roi_coordinates_into_frame():
    box = make_box(1, 0.9, [10, 20, 50, 60], track_id=7)

    detection = process_box(box, CLASS_NAMES, 0.5, ROI, FRAME_W, FRAME_H)

    assert detection["species"] == "zebra"
    assert detection["track_id"] == 7
    assert (detection["x1"], detection["y1"], detection["x2"], detection["y2"]) == (110, 70, 150, 110)
    assert detection["zone"] == "left"


def test_process_box_filters_low_confidence_and_unmapped_classes():
    assert process_box(make_box(1, 0.3, [0, 0, 10, 10]), CLASS_NAMES, 0.5, ROI, FRAME_W, FRAME_H) is None
    assert process_box(make_box(0, 0.9, [0, 0, 10, 10]), CLASS_NAMES, 0.5, ROI, FRAME_W, FRAME_H) is None


def test_process_box_refines_ungulates_and_handles_missing_track_id():
    detection = process_box(make_box(2, 0.9, [0, 0, 10, 30]), CLASS_NAMES, 0.5, ROI, FRAME_W, FRAME_H)

    assert detection["species"] == "antelope_like"
    assert detection["track_id"] is None
