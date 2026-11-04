import os

YOUTUBE_URL = "https://www.youtube.com/watch?v=DsNtwGJXTTs"

# YOLO26 (COCO classes, so CLASS_MAPPING still applies). Downloaded automatically on first run.
# Measured on this Mac with a 640px ROI: yolo26n ~28 fps (CPU), yolo26s ~36 fps (MPS),
# yolo26m ~20 fps (MPS) / ~8 fps (CPU). Medium is noticeably better on small, distant animals;
# drop to yolo26s.pt if running on CPU only. A wildlife-specific model (e.g. MegaDetector or
# a fine-tuned YOLO) would remove the need for the ungulate heuristics below.
MODEL_PATH = "yolo26m.pt"
DEVICE = None  # None = auto: CUDA, else Apple GPU (mps), else CPU. Or "cpu", "mps", "cuda:0".
TRACKER_CONFIG = "bytetrack.yaml"
INFERENCE_SIZE = 640

CONF_THRESHOLD = 0.50
LARGE_GROUP_THRESHOLD = 3
VISIT_GAP_SECONDS = 10

# Frames analysed per second of stream. YouTube live arrives in ~5 s chunks that decode in a
# burst; frames are queued and released at this rate so playback and timestamps stay smooth.
TARGET_FPS = 10
STREAM_MAX_HEIGHT = 720       # lower resolution = less decoding and memory; ROI/zones are relative
MAX_QUEUE_SECONDS = 10        # if analysis falls further behind than this, the oldest frames are dropped
STREAM_TIMEOUT_SECONDS = 15   # give up on a stalled connection and reconnect after this long

# A state change (species appearing/disappearing, large group) must hold for this many
# consecutive frames before an event fires (10 frames = 1 s at TARGET_FPS). Stops flicker spamming events.
EVENT_CONFIRM_FRAMES = 10
EVENT_IMAGE_COOLDOWN_SECONDS = 10

SAVE_ANALYTICS_ON_EXIT = True
SHOW_WINDOW = True

# Stream reconnection
RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_SECONDS = 5

# Region of interest, as fractions of the frame (x1, y1, x2, y2)
ROI_BOUNDS = (0.18, 0.30, 0.82, 0.82)

# Ungulate refinement heuristics (box measured in full-frame pixels)
ANTELOPE_MIN_ASPECT_RATIO = 1.15   # height / width
LARGE_UNGULATE_MIN_AREA = 0.025    # box area / frame area

OUTPUT_DIR = "output"
CHARTS_DIR = "charts"
SAVE_FRAMES = True
FRAMES_DIR = os.path.join(OUTPUT_DIR, "saved_frames")
CSV_FLUSH_EVERY_ROWS = 200

DETECTIONS_CSV = "detections.csv"
FRAME_SUMMARY_CSV = "frame_summary.csv"
EVENTS_CSV = "events.csv"
VISITS_CSV = "visits.csv"
ANALYSIS_SUMMARY_TXT = "analysis_summary.txt"

CLASS_MAPPING = {
    "elephant": "elephant",
    "zebra": "zebra",
    "giraffe": "giraffe",
    "bird": "bird",
    "cow": "ungulate",
    "horse": "ungulate",
}
