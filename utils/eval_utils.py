"""
Scores the detector against hand-reviewed frames (collect_eval_frames.py -> review_app.py -> evaluate.py).

predictions.json  {"frames": [{"id", "file", "timestamp", "width", "height", "roi": [x1, y1, x2, y2],
                               "boxes": [{"raw_class", "species", "conf", "x1", "y1", "x2", "y2"}]}]}
labels.json       {"<frame id>": {"boxes": ["<label for each box, in order>"], "missed": <animals with no box>}}

A reviewer labels every box the model proposed (down to EVAL_MIN_CONF) and counts animals inside the watched
area that got no box at all. That is enough to compute precision and recall at any confidence threshold above
EVAL_MIN_CONF without drawing boxes by hand.
"""
import json
import os
from collections import Counter
from dataclasses import dataclass

from config import ANTELOPE_MIN_ASPECT_RATIO, CLASS_MAPPING, LARGE_UNGULATE_MIN_AREA
from utils.detection_utils import refine_ungulate_label

NOT_ANIMAL = "not_animal"
DUPLICATE = "duplicate"
OTHER_ANIMAL = "other_animal"

# Label -> description shown to the reviewer. The first entries are the pipeline's own species labels.
LABELS = {
    "antelope_like": "Antelope (impala, springbok, kudu, oryx…)",
    "large_ungulate": "Large hoofed animal (buffalo, wildebeest, eland…)",
    "ungulate": "Other hoofed animal, or unsure which",
    "zebra": "Zebra",
    "elephant": "Elephant",
    "giraffe": "Giraffe",
    "bird": "Bird",
    OTHER_ANIMAL: "Another animal (warthog, baboon, predator…)",
    NOT_ANIMAL: "Not an animal (rock, shadow, bush, vehicle…)",
    DUPLICATE: "Second box on an animal that already has one",
}
UNGULATE_LABELS = {"antelope_like", "large_ungulate", "ungulate"}


def read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    # Write to a temporary file first so an interrupted save never leaves half a file behind
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp_path, path)


def is_animal(label):
    return label not in (NOT_ANIMAL, DUPLICATE)


def predicted_species(box, frame):
    return box["species"]


def species_with_thresholds(min_aspect_ratio, min_area):
    """A predicted_species replacement that re-runs the ungulate heuristic with other thresholds."""
    def species(box, frame):
        if CLASS_MAPPING.get(box["raw_class"]) != "ungulate":
            return box["species"]
        return refine_ungulate_label(box["x1"], box["y1"], box["x2"], box["y2"], frame["width"], frame["height"],
                                     min_aspect_ratio, min_area)
    return species


@dataclass
class Score:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    species_correct: int = 0

    @property
    def precision(self):
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else 0.0

    @property
    def recall(self):
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def species_accuracy(self):
        return self.species_correct / self.true_positives if self.true_positives else 0.0


def reviewed(predictions, labels):
    """Yields (frame, box labels, missed count) for every frame that has been reviewed."""
    for frame in predictions["frames"]:
        review = labels.get(frame["id"])
        if review is not None:
            yield frame, review["boxes"], review["missed"]


def score(predictions, labels, conf_threshold, species_of=predicted_species):
    """
    Precision, recall and species accuracy if only boxes at or above conf_threshold were kept.
    A real animal whose box falls below the threshold becomes a miss; duplicates and non-animals that are kept
    count as false alarms.
    """
    result = Score()
    for frame, box_labels, missed in reviewed(predictions, labels):
        result.false_negatives += missed
        for box, label in zip(frame["boxes"], box_labels, strict=True):
            kept = box["conf"] >= conf_threshold
            if is_animal(label):
                if kept:
                    result.true_positives += 1
                    result.species_correct += species_of(box, frame) == label
                else:
                    result.false_negatives += 1
            elif kept:
                result.false_positives += 1
    return result


def confusion(predictions, labels, conf_threshold):
    """Counter of (predicted species, reviewer label) over the boxes kept at conf_threshold."""
    counts = Counter()
    for frame, box_labels, _ in reviewed(predictions, labels):
        for box, label in zip(frame["boxes"], box_labels, strict=True):
            if box["conf"] >= conf_threshold:
                counts[(box["species"], label)] += 1
    return counts


def tune_ungulate_thresholds(predictions, labels, aspect_values, area_values):
    """
    Grid-searches the ungulate heuristic on boxes the model saw as cow/horse and the reviewer labelled as one of
    the ungulate categories. Returns (best aspect, best area, best accuracy, current accuracy, sample count);
    ties go to the values closest to the current settings.
    """
    samples = [
        (box, frame, label)
        for frame, box_labels, _ in reviewed(predictions, labels)
        for box, label in zip(frame["boxes"], box_labels, strict=True)
        if CLASS_MAPPING.get(box["raw_class"]) == "ungulate" and label in UNGULATE_LABELS
    ]
    if not samples:
        return ANTELOPE_MIN_ASPECT_RATIO, LARGE_UNGULATE_MIN_AREA, 0.0, 0.0, 0

    def accuracy(aspect, area):
        species = species_with_thresholds(aspect, area)
        return sum(species(box, frame) == label for box, frame, label in samples) / len(samples)

    def distance_from_current(aspect, area):
        return abs(aspect - ANTELOPE_MIN_ASPECT_RATIO) / ANTELOPE_MIN_ASPECT_RATIO + \
            abs(area - LARGE_UNGULATE_MIN_AREA) / LARGE_UNGULATE_MIN_AREA

    best_aspect, best_area = min(
        ((aspect, area) for aspect in aspect_values for area in area_values),
        key=lambda pair: (-accuracy(*pair), distance_from_current(*pair)),
    )
    return (best_aspect, best_area, accuracy(best_aspect, best_area),
            accuracy(ANTELOPE_MIN_ASPECT_RATIO, LARGE_UNGULATE_MIN_AREA), len(samples))
