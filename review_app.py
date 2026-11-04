"""
Review the frames sampled by collect_eval_frames.py: confirm or correct each box and count missed animals.

    uv run streamlit run review_app.py

Answers are saved to eval/<set>/labels.json after every frame, so you can stop and continue later.
Then run `uv run evaluate.py` for the scores.
"""
import os

import cv2
import streamlit as st

from config import CONF_THRESHOLD, EVAL_DIR
from utils.eval_utils import LABELS, read_json, write_json

LABEL_KEYS = list(LABELS)
KEPT_COLOR = (60, 200, 60)       # BGR: at or above the live pipeline's confidence threshold
DROPPED_COLOR = (0, 190, 255)    # below it: the pipeline ignores these, but they still need a label
ROI_COLOR = (230, 120, 40)

st.set_page_config(page_title="Detection review", page_icon=":material/fact_check:", layout="wide")


def frame_sets():
    if not os.path.isdir(EVAL_DIR):
        return []
    return sorted(name for name in os.listdir(EVAL_DIR) if os.path.exists(os.path.join(EVAL_DIR, name, "predictions.json")))


def annotated(image, frame):
    image = image.copy()
    x1, y1, x2, y2 = frame["roi"]
    cv2.rectangle(image, (x1, y1), (x2, y2), ROI_COLOR, 2)
    for number, box in enumerate(frame["boxes"], start=1):
        color = KEPT_COLOR if box["conf"] >= CONF_THRESHOLD else DROPPED_COLOR
        cv2.rectangle(image, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
        label_origin = (box["x1"], max(box["y1"] - 6, 18))
        cv2.putText(image, str(number), label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(image, str(number), label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop(image, box, height=150):
    """The box with some surroundings, enlarged so small, distant animals are recognisable."""
    pad_x, pad_y = (box["x2"] - box["x1"]) // 2 + 10, (box["y2"] - box["y1"]) // 2 + 10
    h, w = image.shape[:2]
    piece = image[max(box["y1"] - pad_y, 0):min(box["y2"] + pad_y, h), max(box["x1"] - pad_x, 0):min(box["x2"] + pad_x, w)]
    scale = height / max(piece.shape[0], 1)
    piece = cv2.resize(piece, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)


def next_unreviewed(frames, labels, start):
    for i in list(range(start, len(frames))) + list(range(0, start)):
        if frames[i]["id"] not in labels:
            return i
    return None


sets = frame_sets()
if not sets:
    st.title("Detection review")
    st.info("No frames to review yet. Collect some first with `uv run collect_eval_frames.py`.")
    st.stop()

with st.sidebar:
    set_name = st.selectbox("Frame set", sets, key="set_name")
    st.markdown(
        "**How to review**\n\n"
        "For each numbered box, pick what it really is. Green boxes are ones the live pipeline keeps; "
        f"amber ones fall below its confidence threshold ({CONF_THRESHOLD}) but still need an answer.\n\n"
        "Then count animals **inside the blue area** that have no box at all, and save."
    )

set_dir = os.path.join(EVAL_DIR, set_name)
labels_path = os.path.join(set_dir, "labels.json")
frames = read_json(os.path.join(set_dir, "predictions.json"), {"frames": []})["frames"]
labels = read_json(labels_path, {})

if st.session_state.get("index_for") != set_name:
    st.session_state.index_for = set_name
    st.session_state.index = next_unreviewed(frames, labels, 0) or 0

index = st.session_state.index
frame = frames[index]
reviewed_count = sum(f["id"] in labels for f in frames)

st.title("Detection review")
st.progress(reviewed_count / len(frames), text=f"Reviewed {reviewed_count} of {len(frames)} frames")

nav = st.columns([1, 1, 2, 4])
if nav[0].button("Previous", disabled=index == 0, width="stretch"):
    st.session_state.index -= 1
    st.rerun()
if nav[1].button("Next", disabled=index == len(frames) - 1, width="stretch"):
    st.session_state.index += 1
    st.rerun()
pending = next_unreviewed(frames, labels, index)
if nav[2].button("Next unreviewed", disabled=pending is None or pending == index, width="stretch"):
    st.session_state.index = pending
    st.rerun()
status = "reviewed" if frame["id"] in labels else "not reviewed yet"
nav[3].markdown(f"Frame **{index + 1}** of {len(frames)} · {frame['timestamp']} · {status}")

if pending is None:
    st.success("All frames are reviewed. Run `uv run evaluate.py` for the scores.")

image = cv2.imread(os.path.join(set_dir, frame["file"]))
if image is None:
    st.error(f"Image {frame['file']} is missing from {set_dir}/. Frames aren't stored in git; collect them on this machine.")
    st.stop()

existing = labels.get(frame["id"])
picture, answers = st.columns([3, 2])
picture.image(annotated(image, frame), width="stretch")
picture.caption("Blue: the area the model watches. Numbers match the list on the right.")

with answers.form(key=f"review-{set_name}-{frame['id']}", border=False):
    choices = []
    if not frame["boxes"]:
        st.write("The model proposed no boxes in this frame.")
    for number, box in enumerate(frame["boxes"], start=1):
        crop_column, choice_column = st.columns([1, 2], vertical_alignment="center")
        crop_column.image(crop(image, box))
        default = existing["boxes"][number - 1] if existing else box["species"]
        kept = "kept" if box["conf"] >= CONF_THRESHOLD else "below threshold"
        choices.append(choice_column.selectbox(
            f"Box {number}: model says {box['species'].replace('_', ' ')} ({box['conf']:.2f}, {kept})",
            LABEL_KEYS, index=LABEL_KEYS.index(default), format_func=LABELS.get,
            key=f"{set_name}-{frame['id']}-{number}",
        ))
    missed = st.number_input(
        "Animals inside the blue area with no box", min_value=0, step=1,
        value=existing["missed"] if existing else 0, key=f"{set_name}-{frame['id']}-missed",
    )
    if st.form_submit_button("Save and next", type="primary"):
        labels[frame["id"]] = {"boxes": choices, "missed": int(missed)}
        write_json(labels_path, labels)
        following = next_unreviewed(frames, labels, index)
        st.session_state.index = following if following is not None else min(index + 1, len(frames) - 1)
        st.rerun()
