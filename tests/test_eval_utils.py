import pytest

from utils.eval_utils import (
    DUPLICATE, NOT_ANIMAL, confusion, read_json, score, tune_ungulate_thresholds, write_json,
)


def box(species, conf, raw_class=None, x1=0, y1=0, x2=10, y2=10):
    return {"species": species, "raw_class": raw_class or species, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def frame(frame_id, *boxes):
    return {"id": frame_id, "width": 1000, "height": 1000, "boxes": list(boxes)}


PREDICTIONS = {"frames": [
    frame("a", box("zebra", 0.9), box("zebra", 0.4), box("bird", 0.6)),
    frame("b", box("elephant", 0.8), box("elephant", 0.7)),
    frame("c"),
    frame("unreviewed", box("zebra", 0.9)),
]}
LABELS = {
    "a": {"boxes": ["zebra", "zebra", NOT_ANIMAL], "missed": 1},
    "b": {"boxes": ["giraffe", DUPLICATE], "missed": 0},
    "c": {"boxes": [], "missed": 2},
}


def test_score_at_threshold():
    result = score(PREDICTIONS, LABELS, conf_threshold=0.5)

    # kept: zebra 0.9 (right), bird 0.6 (not an animal), elephant 0.8 (an animal, wrong species), duplicate 0.7
    assert result.true_positives == 2
    assert result.false_positives == 2
    # missed: 1 + 2 counted by the reviewer, plus the real zebra at 0.4 dropped by the threshold
    assert result.false_negatives == 4
    assert result.species_correct == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(2 / 6)
    assert result.species_accuracy == pytest.approx(0.5)


def test_lower_threshold_trades_precision_for_recall():
    low, high = score(PREDICTIONS, LABELS, 0.3), score(PREDICTIONS, LABELS, 0.85)

    assert low.recall > high.recall
    assert high.precision == 1.0


def test_confusion_counts_kept_boxes():
    counts = confusion(PREDICTIONS, LABELS, 0.5)

    assert counts[("elephant", "giraffe")] == 1
    assert counts[("bird", NOT_ANIMAL)] == 1
    assert counts[("zebra", "zebra")] == 1  # the 0.4 zebra is below the threshold


def test_tune_ungulate_thresholds_finds_the_separating_aspect_ratio():
    # Antelope boxes are clearly taller than wide (ratio 1.5); other hoofed animals are wider (ratio 0.6)
    tall = [box("ungulate", 0.9, "cow", 0, 0, 20, 30) for _ in range(5)]
    wide = [box("ungulate", 0.9, "horse", 0, 0, 50, 30) for _ in range(5)]
    predictions = {"frames": [frame("f", *tall, *wide)]}
    labels = {"f": {"boxes": ["antelope_like"] * 5 + ["ungulate"] * 5, "missed": 0}}

    aspect, area, best, current, n = tune_ungulate_thresholds(predictions, labels, [0.5, 1.0, 1.4, 2.0], [0.5])

    assert n == 10
    assert best == 1.0
    assert 0.6 < aspect <= 1.5


def test_tune_ungulate_thresholds_without_samples():
    assert tune_ungulate_thresholds(PREDICTIONS, LABELS, [1.0], [0.01])[4] == 0


def test_write_json_round_trip(tmp_path):
    path = str(tmp_path / "labels.json")
    write_json(path, LABELS)

    assert read_json(path, {}) == LABELS
    assert read_json(str(tmp_path / "missing.json"), {"x": 1}) == {"x": 1}
