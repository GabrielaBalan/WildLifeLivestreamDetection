from config import CLASS_MAPPING, ANTELOPE_MIN_ASPECT_RATIO, LARGE_UNGULATE_MIN_AREA


def get_zone(cx: float, x_start: float, x_end: float) -> str:
    """Splits the horizontal span [x_start, x_end) into left/center/right thirds."""
    third = (x_end - x_start) / 3
    if cx < x_start + third:
        return "left"
    elif cx < x_start + 2 * third:
        return "center"
    return "right"


def refine_ungulate_label(x1: int, y1: int, x2: int, y2: int, frame_width: int, frame_height: int) -> str:
    width = x2 - x1
    height = y2 - y1
    area = width * height

    aspect_ratio = height / max(width, 1)
    relative_area = area / max(frame_width * frame_height, 1)

    if aspect_ratio >= ANTELOPE_MIN_ASPECT_RATIO:
        return "antelope_like"

    if relative_area >= LARGE_UNGULATE_MIN_AREA:
        return "large_ungulate"

    return "ungulate"


def process_box(box, class_names, conf_threshold, roi_bounds, frame_width, frame_height):
    """
    Converts one YOLO box (in ROI pixel coordinates) to a full-frame detection dict.
    Returns None for low-confidence or unmapped classes.
    """
    conf = float(box.conf[0])
    if conf < conf_threshold:
        return None

    raw_class = class_names[int(box.cls[0])]
    if raw_class not in CLASS_MAPPING:
        return None

    roi_x1, roi_y1, roi_x2, _ = roi_bounds
    rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])

    x1 = roi_x1 + rx1
    y1 = roi_y1 + ry1
    x2 = roi_x1 + rx2
    y2 = roi_y1 + ry2

    species = CLASS_MAPPING[raw_class]
    if species == "ungulate":
        species = refine_ungulate_label(x1, y1, x2, y2, frame_width, frame_height)

    track_id = int(box.id[0]) if box.id is not None else None

    return {
        "species": species,
        "raw_class": raw_class,
        "conf": conf,
        "track_id": track_id,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "zone": get_zone((x1 + x2) / 2, roi_x1, roi_x2),
    }
