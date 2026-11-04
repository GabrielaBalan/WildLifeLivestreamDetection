"""
Scores the detector against the frames reviewed in review_app.py and suggests settings.

    uv run evaluate.py [--set default]

Prints the report and saves it as eval/<set>/report.md.
"""
import argparse
import os

import numpy as np

from config import ANTELOPE_MIN_ASPECT_RATIO, CONF_THRESHOLD, EVAL_DIR, EVAL_MIN_CONF, LARGE_UNGULATE_MIN_AREA
from utils.eval_utils import LABELS, confusion, read_json, reviewed, score, tune_ungulate_thresholds

# Fewer reviewed boxes than this and a suggested change is mostly noise
MIN_SAMPLES_FOR_ADVICE = 30


def pct(value):
    return f"{value:.0%}"


def main():
    parser = argparse.ArgumentParser(description="Score the detector against reviewed frames.")
    parser.add_argument("--set", default="default", help="name of the frame set (a folder under eval/)")
    args = parser.parse_args()

    set_dir = os.path.join(EVAL_DIR, args.set)
    predictions = read_json(os.path.join(set_dir, "predictions.json"), None)
    labels = read_json(os.path.join(set_dir, "labels.json"), {})
    if predictions is None:
        print(f"No frames in {set_dir}/. Collect some first: uv run collect_eval_frames.py --set {args.set}")
        return
    frames = list(reviewed(predictions, labels))
    if not frames:
        print(f"No reviewed frames in {set_dir}/ yet. Review them first: uv run streamlit run review_app.py")
        return

    report = []

    def out(line=""):
        print(line)
        report.append(line)

    box_count = sum(len(frame["boxes"]) for frame, _, _ in frames)
    out(f"# Detector evaluation: {args.set}")
    out()
    out(f"{len(frames)} reviewed frames ({len(predictions['frames'])} collected), {box_count} boxes, "
        f"{sum(missed for _, _, missed in frames)} animals with no box.")

    current = score(predictions, labels, CONF_THRESHOLD)
    out()
    out(f"## Current settings (confidence threshold {CONF_THRESHOLD})")
    out()
    out(f"- **Precision** {pct(current.precision)}: share of kept boxes that really are animals "
        f"({current.true_positives} of {current.true_positives + current.false_positives})")
    out(f"- **Recall** {pct(current.recall)}: share of animals that get a kept box "
        f"({current.true_positives} of {current.true_positives + current.false_negatives})")
    out(f"- **Species accuracy** {pct(current.species_accuracy)}: share of found animals with the right label "
        f"({current.species_correct} of {current.true_positives})")

    mistakes = [(pair, n) for pair, n in confusion(predictions, labels, CONF_THRESHOLD).most_common() if pair[0] != pair[1]]
    if mistakes:
        out()
        out("Most common mistakes among kept boxes:")
        out()
        out("| Model said | Reviewer said | Boxes |")
        out("|---|---|---:|")
        for (said, truth), n in mistakes[:8]:
            out(f"| {said} | {LABELS[truth]} | {n} |")

    out()
    out("## Confidence threshold")
    out()
    out("| Threshold | Precision | Recall | F1 | Kept boxes | False alarms | Missed |")
    out("|---:|---:|---:|---:|---:|---:|---:|")
    thresholds = np.round(np.arange(EVAL_MIN_CONF, 0.901, 0.05), 2)
    sweep = [(t, score(predictions, labels, t)) for t in thresholds]
    best_t, best = max(sweep, key=lambda item: (item[1].f1, -abs(item[0] - CONF_THRESHOLD)))
    for t, s in sweep:
        marker = " (current)" if np.isclose(t, CONF_THRESHOLD) else ""
        marker += " (best F1)" if t == best_t else ""
        out(f"| {t:.2f}{marker} | {pct(s.precision)} | {pct(s.recall)} | {s.f1:.2f} | "
            f"{s.true_positives + s.false_positives} | {s.false_positives} | {s.false_negatives} |")

    out()
    animal_boxes = current.true_positives + current.false_negatives
    if animal_boxes < MIN_SAMPLES_FOR_ADVICE:
        out(f"Only {animal_boxes} animals reviewed so far; review more frames before changing the threshold.")
    elif np.isclose(best_t, CONF_THRESHOLD):
        out(f"The current threshold ({CONF_THRESHOLD}) already gives the best balance (F1).")
    else:
        out(f"Suggestion: `CONF_THRESHOLD = {best_t:.2f}` in config.py (F1 {current.f1:.2f} -> {best.f1:.2f}).")

    out()
    out("## Ungulate heuristic (antelope / large / other hoofed)")
    out()
    # Include the current values so a tie keeps them rather than nudging to a neighbouring grid point
    aspect_values = np.union1d(np.round(np.arange(0.4, 2.01, 0.05), 2), [ANTELOPE_MIN_ASPECT_RATIO])
    area_values = np.union1d(np.round(np.geomspace(0.002, 0.1, 25), 4), [LARGE_UNGULATE_MIN_AREA])
    aspect, area, best_accuracy, current_accuracy, n = tune_ungulate_thresholds(
        predictions, labels, aspect_values, area_values)
    if n == 0:
        out("No reviewed boxes of hoofed animals yet.")
    else:
        out(f"{n} reviewed boxes the model saw as cow/horse and the reviewer called antelope, large or other hoofed.")
        out()
        out(f"- Current: aspect ratio >= {ANTELOPE_MIN_ASPECT_RATIO}, area >= {LARGE_UNGULATE_MIN_AREA}: "
            f"{pct(current_accuracy)} labelled right")
        out(f"- Best found: aspect ratio >= {aspect}, area >= {area}: {pct(best_accuracy)} labelled right")
        out()
        if n < MIN_SAMPLES_FOR_ADVICE:
            out(f"Only {n} samples; review more frames with hoofed animals before changing these.")
        elif best_accuracy - current_accuracy < 0.05:
            out("The current values are about as good as anything found; no change needed.")
        else:
            out(f"Suggestion: `ANTELOPE_MIN_ASPECT_RATIO = {aspect}` and `LARGE_UNGULATE_MIN_AREA = {area}` in config.py.")

    report_path = os.path.join(set_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nSaved {report_path}")


if __name__ == "__main__":
    main()
