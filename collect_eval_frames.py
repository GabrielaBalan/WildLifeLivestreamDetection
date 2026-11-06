"""
Samples frames from the livestream for the accuracy evaluation.

    uv run collect_eval_frames.py                      # 150 frames, one every 15 s (~40 min)
    uv run collect_eval_frames.py --count 50 --every 30

Saves each full frame plus every box the model proposes down to EVAL_MIN_CONF, so evaluate.py can test higher
confidence thresholds too. Running it again adds to the same set. Review the frames afterwards with
`uv run streamlit run review_app.py`.
"""
import argparse
import logging
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from config import EVAL_DIR, EVAL_MIN_CONF, EVAL_SAMPLE_SECONDS, INFERENCE_SIZE, MODEL_PATH, YOUTUBE_URL
from main import select_device
from utils.detection_utils import process_box
from utils.eval_utils import read_json, write_json
from utils.logging_utils import setup_logging
from utils.stream_utils import StreamReader, format_timestamp, get_roi

BOX_FIELDS = ("raw_class", "species", "x1", "y1", "x2", "y2")

log = logging.getLogger("collect")


def main():
    parser = argparse.ArgumentParser(description="Sample livestream frames for the accuracy evaluation.")
    parser.add_argument("--set", default="default", help="name of the frame set (a folder under eval/)")
    parser.add_argument("--count", type=int, default=150, help="number of frames to collect (default: 150)")
    parser.add_argument("--every", type=float, default=EVAL_SAMPLE_SECONDS,
                        help=f"seconds between samples (default: {EVAL_SAMPLE_SECONDS})")
    args = parser.parse_args()

    set_dir = os.path.join(EVAL_DIR, args.set)
    setup_logging(os.path.join(set_dir, "collect.log"))
    os.makedirs(os.path.join(set_dir, "frames"), exist_ok=True)
    predictions_path = os.path.join(set_dir, "predictions.json")
    predictions = read_json(predictions_path, {"frames": []})

    device = select_device()
    log.info(f"Loading YOLO model {MODEL_PATH} on {device}...")
    model = YOLO(MODEL_PATH)
    reader = StreamReader(YOUTUBE_URL).start()

    collected = 0
    next_sample = 0.0
    log.info(f"Collecting {args.count} frames, one every {args.every:g} s. Press Ctrl+C to stop early.")

    try:
        while collected < args.count:
            frame = reader.read()
            if frame is None:
                if reader.finished:
                    log.info("The stream has ended.")
                    break
                continue

            if time.monotonic() < next_sample:
                continue
            next_sample = time.monotonic() + args.every

            now = datetime.now()
            frame_id = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            frame_height, frame_width = frame.shape[:2]
            roi, roi_bounds = get_roi(frame)

            results = model(roi, imgsz=INFERENCE_SIZE, conf=EVAL_MIN_CONF, device=device, verbose=False)[0]
            boxes = []
            for box in results.boxes:
                detection = process_box(box, model.names, EVAL_MIN_CONF, roi_bounds, frame_width, frame_height)
                if detection is not None:
                    boxes.append({field: detection[field] for field in BOX_FIELDS} | {"conf": round(detection["conf"], 4)})

            file = f"frames/{frame_id}.jpg"
            cv2.imwrite(os.path.join(set_dir, file), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            predictions["frames"].append({
                "id": frame_id, "file": file, "timestamp": format_timestamp(now),
                "width": frame_width, "height": frame_height, "roi": list(roi_bounds), "boxes": boxes,
            })
            write_json(predictions_path, predictions)

            collected += 1
            log.info(f"[{collected}/{args.count}] {format_timestamp(now)}: {len(boxes)} candidate boxes")

    except KeyboardInterrupt:
        log.info("Interrupted.")

    finally:
        reader.stop()

    log.info(f"{len(predictions['frames'])} frames in {set_dir}/. Review them with: uv run streamlit run review_app.py")


if __name__ == "__main__":
    main()
