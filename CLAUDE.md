# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Wildlife detection on a YouTube livestream (an Africam waterhole camera in South Africa): YOLO26 + ByteTrack find and
track animals, the live loop logs detections/visits/events to CSV, and separate tools chart, dashboard and evaluate
the results. Developed on macOS (Apple GPU via MPS); the user works in VS Code.

## Commands

Dependencies are managed with uv (`pyproject.toml` + `uv.lock`, Python 3.12, venv in `.venv/`). The user's shell may
still export `VIRTUAL_ENV` for an old PyCharm venv; `unset VIRTUAL_ENV` before `uv run` avoids the mismatch warning.

```bash
uv sync                                         # install / update the environment
uv run pytest                                   # all tests
uv run pytest tests/test_eval_utils.py::test_score_at_threshold   # a single test
uv run ruff check .                             # lint (rules pinned in pyproject.toml)

uv run main.py                                  # live pipeline -> output/ ('q' in the window or Ctrl+C to stop)
uv run streamlit run dashboard.py               # live dashboard (port 8501), refreshes while main.py runs
uv run analyze_data.py [--data-dir docs]        # matplotlib charts -> charts/, summary txt
uv run export_dashboard.py --data-dir docs      # self-contained HTML dashboard -> docs/dashboard.html

uv run collect_eval_frames.py                   # accuracy eval 1/3: sample frames -> eval/<set>/
uv run streamlit run review_app.py --server.port 8502   # 2/3: human review -> eval/<set>/labels.json
uv run evaluate.py                              # 3/3: precision/recall + suggested thresholds -> eval/<set>/report.md
```

`ruff format` is not enforced; don't reformat the whole repo. Add dependencies with `uv add`.

CI (`.github/workflows/ci.yml`) runs ruff and pytest after `uv sync --only-group test`, which installs only pytest,
headless OpenCV, pandas and matplotlib. **Tests must not import ultralytics, torch, streamlit or `main.py`**, or CI
breaks. Test pure logic (tracking, events, eval scoring, stream pacing with the queue fed directly) instead.

## Architecture

**Live loop (`main.py`)**: `StreamReader` → ROI crop → `model.track` → `process_box` → stats, visits, events → CSV.

- `utils/stream_utils.StreamReader` gets an HLS URL from yt-dlp (video-only, ≤720p; YouTube no longer offers combined
  formats) and decodes on a background thread. YouTube live arrives in ~5 s chunks that decode in a burst, so frames
  are queued with their stream position and released on a playback clock at `TARGET_FPS` (10). On queue overflow it
  skips ahead and restarts the clock. `read()` returns `None` after ≤50 ms so the main loop can keep calling
  `cv2.waitKey`; never make it block, or the window freezes and Ctrl+C stops working on macOS.
- The ROI is passed to YOLO directly (it letterboxes internally), so boxes are in ROI pixels and `process_box`
  offsets them by the ROI origin. Don't add a manual resize: an earlier version did and stored unscaled coordinates.
- `CLASS_MAPPING` maps COCO classes to species; cow/horse become `antelope_like` / `large_ungulate` / `ungulate` by
  box shape (`refine_ungulate_label`, thresholds in config, tuned by `evaluate.py`). Zones are thirds of the ROI.
- `utils/event_utils.EventDetector` debounces per-species presence (`EVENT_CONFIRM_FRAMES`) before raising events.
  `tracking.VisitTracker` closes a species' visit after `VISIT_GAP_SECONDS` unseen.
- `utils/csv_utils.SessionLogs` appends to `output/*.csv` (buffered) and holds an `fcntl` lock on the folder, so a
  second `main.py` exits. Cleanup (open visits, session summary) runs in `main`'s `finally`.

**Data contract**: every row carries `session_id` + `frame_id`; timestamps are **UTC** with milliseconds and a
trailing `Z` (`format_timestamp`); list/dict columns are JSON. In `detections.csv`, `species` is the tracked
animal's confidence-weighted majority label (`tracking.TrackSpecies`) and `frame_species` the per-frame guess.
`health.csv` gets a row every `HEALTH_LOG_SECONDS` (fps, inference ms, dropped frames, reconnects) and `run.log`
the log messages. `docs/` holds archived runs in an **older format** (Python-repr columns, no session IDs, naive
local times of the recording Mac, recorded before the coordinate and event-flicker fixes), so readers must accept
both. `analyze_data.load_data` is the one shared loader: it parses either format, converts all times to naive
**camera local time** (`to_camera_time`, `CAMERA_TIMEZONE` / `LEGACY_LOG_TIMEZONE`), and skips repeated header rows
and unparseable rows. Zone usage and co-occurrence are derived from the logs, never stored.

**Analysis and dashboards**: `dashboard_data.build_tables` aggregates to minute-level tables (sessions inferred from
>2 min gaps when there's no `session_id`). Both dashboards read from it, so their numbers always agree:
`dashboard.py` (Streamlit, refreshes via `st.fragment(run_every=…)`, cache keyed by file mtimes) and
`export_dashboard.py`, which injects the tables as JSON into `dashboard_template.html` at `__DASHBOARD_DATA__`. The
template is a self-contained page with vanilla-JS SVG charts and light/dark themes; it's published as a claude.ai
artifact ("Savanna Cam Log").

**Accuracy evaluation**: `collect_eval_frames.py` saves full frames (gitignored) and every box down to
`EVAL_MIN_CONF` 0.25 (plain predict, no tracking) to `predictions.json`. The reviewer labels every box and counts
animals inside the ROI with no box, which is enough for `utils/eval_utils.score` to compute precision and recall at
any threshold ≥ 0.25 without drawing ground-truth boxes. The first seven keys of `LABELS` are the pipeline's species.

**Config**: everything lives in `config.py` as constants imported by name (`from config import X`), so overriding a
value means editing the file (or a copy) before import. Changing thresholds mid-collection makes sessions
incomparable; keep config fixed during a data-collection period.

## Working on this repo

- Smoke-test the live loop in a copy of the code (not the repo's `output/`, which may be locked by the user's run),
  with `SHOW_WINDOW = False`. A process started with `&` from a non-interactive shell ignores SIGINT; restore it with
  `signal.signal(signal.SIGINT, signal.default_int_handler)` before `runpy.run_path("main.py", run_name="__main__")`
  to test the Ctrl+C path.
- At night the camera switches to infrared and the scene is usually empty; check the time at the camera (UTC+2)
  before judging detection results.
- `update` is the default branch and acts as main; there is no `main` branch.
