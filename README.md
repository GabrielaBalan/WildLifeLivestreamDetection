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
Livestream → Frame processing → ROI selection → YOLO detection → Feature extraction → Event generation → CSV storage → Analysis charts

## 4. Extracted Information
The system extracts:
- animal category
- confidence score
- bounding box
- center point
- zone
- timestamp
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

## 6. Data Storage
Data is stored in CSV files:
- detections.csv
- frame_summary.csv
- events.csv
- visits.csv
- zone_usage.csv
- co_occurrence.csv

## 7. Analysis
Charts were generated to show:
- detections per species
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

## 10. How to Run
```bash
pip install -r requirements.txt
python main.py
python analyze_data.py
