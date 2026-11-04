# Wildlife Livestream Monitoring with Python and OpenCV

## 1. Project Goal
This project analyzes a live wildlife camera stream using Python, OpenCV, and YOLO. The system detects animals in real time, extracts structured information, generates events, stores results in CSV files, and produces analysis charts.

## 2. Technology
- Python
- OpenCV
- YOLO / Ultralytics
- Pandas
- Matplotlib
- CSV storage

## 3. Pipeline
Livestream → Frame processing → ROI selection → YOLO detection + ByteTrack tracking → Feature extraction → Debounced event generation → CSV storage → Analysis charts

| File | Role |
|---|---|
| `main.py` | Live loop: read stream, detect + track, log, raise events, show window |
| `config.py` | All settings (stream URL, model, thresholds, ROI, output paths) |
| `tracking.py` | `VisitTracker` – groups detections into per-species visits |
| `utils/stream_utils.py` | yt-dlp URL lookup, background frame reader with reconnects, ROI, snapshots |
| `utils/detection_utils.py` | Converts YOLO boxes to frame-space detections, zones, ungulate heuristics |
| `utils/event_utils.py` | Debounced scene events (activity start/end, new species, groups) |
| `utils/csv_utils.py` | Buffered CSV logs for detections, frames, events, visits |
| `utils/analytics_utils.py` | On-screen overlay and per-session summary |
| `analyze_data.py` | Offline charts and summary from the CSV logs |
| `dashboard_data.py` | Minute-level aggregates shared by both dashboards |
| `dashboard.py` | Live Streamlit dashboard that refreshes while `main.py` runs |
| `export_dashboard.py` + `dashboard_template.html` | Self-contained HTML dashboard snapshot for sharing |

## 4. Extracted Information
The system extracts:
- animal category
- confidence score
- bounding box
- tracker ID (distinct individuals)
- center point
- zone (left / center / right third of the ROI)
- timestamp (millisecond resolution)
- total animals per frame
- visit duration
- co-occurrence

## 5. Events
Generated events include:
- activity_started
- activity_ended
- new_species_detected
- multi_species_detected
- large_group_detected

A state change must hold for `EVENT_CONFIRM_FRAMES` consecutive frames before an event fires, and event snapshots are rate-limited by `EVENT_IMAGE_COOLDOWN_SECONDS`.

## 6. Data Storage
Each run appends to CSV files in `output/` (every row carries a `session_id` and `frame_id`):
- detections.csv
- frame_summary.csv
- events.csv
- visits.csv
- session_summary_<session_id>.txt

Zone usage and co-occurrence are computed from these files by `analyze_data.py`. Results from earlier runs (older CSV format) are kept in `docs/`.

## 7. Analysis
Charts were generated to show:
- detections per species
- distinct individuals per species
- activity over time
- event distribution
- confidence by species
- visit duration
- zone usage

## 8. IoT / Big Data Use
The system can act as an edge AI component. Instead of sending raw video, it outputs lightweight structured events. These events can be sent through MQTT, stored in a database, or ingested into a Big Data platform for dashboards and long-term analysis.

## 9. Limitations
- The model is limited to supported YOLO classes.
- Small or distant animals are harder to detect.
- Exact behavior detection is future work.
- Some categories such as antelope_like are heuristic-based.
- Tracker IDs can change after an occlusion, so individual counts are an upper bound.

## 10. How to Run
Setup uses [uv](https://docs.astral.sh/uv/) (`brew install uv`). It installs Python 3.12 and the exact package
versions from `uv.lock` into `.venv/` inside the project:

```bash
uv sync
```

Then:

```bash
uv run main.py                          # press 'q' in the window or Ctrl+C to stop
uv run analyze_data.py                  # charts from output/
uv run analyze_data.py --data-dir docs  # charts from the archived results
uv run pytest                           # unit tests
uv run ruff check .                     # lint
```

Only one `main.py` can log to `output/` at a time; a second one stops with a message instead of mixing up the files.

### Dashboards
```bash
uv run streamlit run dashboard.py               # live dashboard; run in a second terminal next to main.py
uv run export_dashboard.py --data-dir docs      # writes docs/dashboard.html, a single file you can open or share
```
Both have a day filter, a species filter, and a table view for every chart.

Set `SHOW_WINDOW = False` in `config.py` to run headless (e.g. on a server or edge device). Use
`caffeinate -i uv run main.py` to keep a Mac awake during long runs.

### VS Code
Select the interpreter in `.venv/` (**Python: Select Interpreter**). Run and Debug has entries for the live
detection, the dashboard, and both together; **Terminal → Run Task** starts them without the debugger, which is faster
for long runs.

### Continuous integration
GitHub Actions (`.github/workflows/ci.yml`) runs `ruff` and the tests on every push and pull request. It installs only
the light `test` dependency group, so no PyTorch or model download is needed.
