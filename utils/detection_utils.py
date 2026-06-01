from config import CLASS_MAPPING


def get_zone(cx: float, frame_width: int) -> str:
    if cx < frame_width / 3:
        return "left"
    elif cx < 2 * frame_width / 3:
        return "center"
    return "right"


def refine_ungulate_label(x1: int, y1: int, x2: int, y2: int, frame_width: int, frame_height: int) -> str:
    width = x2 - x1
    height = y2 - y1
    area = width * height

    aspect_ratio = height / max(width, 1)
    relative_area = area / max(frame_width * frame_height, 1)

    if aspect_ratio >= 1.15:
        return "antelope_like"

    if relative_area >= 0.025:
        return "large_ungulate"

    return "ungulate"


def process_box(box, model, conf_threshold, roi_x1, roi_y1, frame_width, frame_height):
    conf = float(box.conf[0])
    if conf < conf_threshold:
        return None

    cls_id = int(box.cls[0])
    raw_class = model.names[cls_id]

    if raw_class not in CLASS_MAPPING:
        return None

    rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])

    x1 = roi_x1 + rx1
    y1 = roi_y1 + ry1
    x2 = roi_x1 + rx2
    y2 = roi_y1 + ry2

    species = CLASS_MAPPING[raw_class]
    if species == "ungulate":
        species = refine_ungulate_label(x1, y1, x2, y2, frame_width, frame_height)

    cx = (x1 + x2) / 2
    zone = get_zone(cx, frame_width)

    return {
        "species": species,
        "raw_class": raw_class,
        "conf": conf,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "zone": zone,
    }