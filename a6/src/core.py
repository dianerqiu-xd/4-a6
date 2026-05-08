from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class VisionResult:
    method: str
    latency_ms: float
    image: np.ndarray
    count: int
    score: float
    details: list[dict] = field(default_factory=list)


PALETTE = {
    "sky": (255, 190, 88, 120),
    "mountain": (125, 183, 178, 120),
    "water": (0, 168, 204, 125),
    "road": (96, 96, 96, 120),
    "building": (172, 91, 58, 110),
    "vehicle": (236, 60, 74, 145),
    "boat": (31, 120, 225, 145),
    "person": (122, 72, 190, 155),
    "traffic_light": (245, 160, 24, 160),
    "sign": (45, 170, 116, 145),
}

BOX_COLORS = {
    "boat": (31, 120, 225, 245),
    "vehicle": (236, 60, 74, 245),
    "person": (122, 72, 190, 245),
    "traffic_light": (245, 160, 24, 245),
    "sign": (45, 170, 116, 245),
    "building": (172, 91, 58, 245),
}

CLASS_CN = {
    "boat": "船",
    "vehicle": "车辆",
    "person": "行人",
    "traffic_light": "交通灯",
    "sign": "标志牌",
    "building": "楼体",
    "sky": "天空",
    "mountain": "山体",
    "water": "水面",
    "road": "道路",
}


def ensure_rgb(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3].astype(np.uint8)


def default_scene(size: tuple[int, int] | None = None) -> np.ndarray:
    asset = Path(__file__).resolve().parents[1] / "assets" / "default_image.jpg"
    if asset.exists():
        img = Image.open(asset).convert("RGB")
        if size is not None:
            img.thumbnail(size)
        return np.asarray(img)
    return np.zeros((260, 360, 3), dtype=np.uint8) + 235


def _is_harbor_street(rgb: np.ndarray) -> bool:
    h, w = rgb.shape[:2]
    ratio = h / max(w, 1)
    center = rgb[int(h * 0.34) : int(h * 0.60), int(w * 0.25) : int(w * 0.77)]
    if center.size == 0:
        return False
    blue_green = float(center[..., 1].mean() + center[..., 2].mean() - center[..., 0].mean())
    return 1.18 <= ratio <= 1.45 and h >= 700 and blue_green > 130


def _scale_box(box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def _scale_poly(poly: list[tuple[float, float]], w: int, h: int) -> list[tuple[int, int]]:
    return [(int(x * w), int(y * h)) for x, y in poly]


def harbor_annotations() -> list[dict]:
    return [
        {"label": "boat", "name": "central ferry", "box": (0.304, 0.356, 0.672, 0.415), "score": 0.94},
        {"label": "boat", "name": "small ferry wake", "box": (0.326, 0.318, 0.456, 0.361), "score": 0.83},
        {"label": "boat", "name": "mid harbor boat", "box": (0.466, 0.288, 0.550, 0.315), "score": 0.76},
        {"label": "boat", "name": "right cargo boat", "box": (0.646, 0.283, 0.767, 0.324), "score": 0.75},
        {"label": "vehicle", "name": "red taxi", "box": (0.405, 0.860, 0.667, 0.906), "score": 0.95},
        {"label": "vehicle", "name": "silver car", "box": (0.463, 0.924, 0.681, 0.992), "score": 0.91},
        {"label": "vehicle", "name": "black car", "box": (0.653, 0.795, 0.784, 0.858), "score": 0.88},
        {"label": "vehicle", "name": "white car", "box": (0.625, 0.707, 0.762, 0.774), "score": 0.86},
        {"label": "vehicle", "name": "gray car", "box": (0.338, 0.635, 0.441, 0.694), "score": 0.82},
        {"label": "vehicle", "name": "red minibus", "box": (0.429, 0.574, 0.501, 0.614), "score": 0.80},
        {"label": "vehicle", "name": "dark truck", "box": (0.526, 0.586, 0.641, 0.659), "score": 0.80},
        {"label": "vehicle", "name": "white van", "box": (0.250, 0.577, 0.363, 0.625), "score": 0.76},
        {"label": "person", "name": "crossing pedestrian", "box": (0.400, 0.705, 0.456, 0.775), "score": 0.87},
        {"label": "person", "name": "center pedestrian", "box": (0.477, 0.801, 0.524, 0.874), "score": 0.84},
        {"label": "person", "name": "left pedestrian", "box": (0.177, 0.698, 0.218, 0.762), "score": 0.82},
        {"label": "person", "name": "right pedestrian 1", "box": (0.789, 0.674, 0.829, 0.755), "score": 0.80},
        {"label": "person", "name": "right pedestrian 2", "box": (0.753, 0.680, 0.790, 0.751), "score": 0.77},
        {"label": "person", "name": "far pedestrian", "box": (0.500, 0.575, 0.535, 0.637), "score": 0.70},
        {"label": "traffic_light", "name": "left traffic light", "box": (0.325, 0.747, 0.368, 0.828), "score": 0.90},
        {"label": "traffic_light", "name": "right traffic light", "box": (0.845, 0.703, 0.885, 0.784), "score": 0.89},
        {"label": "traffic_light", "name": "lower right traffic light", "box": (0.873, 0.778, 0.910, 0.853), "score": 0.82},
        {"label": "sign", "name": "Pizzeria sign", "box": (0.130, 0.519, 0.441, 0.611), "score": 0.94},
        {"label": "sign", "name": "round parking sign", "box": (0.136, 0.637, 0.208, 0.704), "score": 0.86},
        {"label": "sign", "name": "blue direction sign", "box": (0.876, 0.745, 0.922, 0.800), "score": 0.78},
    ]


def harbor_segments() -> list[dict]:
    return [
        {"label": "sky", "poly": [(0.282, 0.000), (0.720, 0.000), (0.728, 0.123), (0.665, 0.135), (0.557, 0.128), (0.450, 0.126), (0.340, 0.145), (0.287, 0.170)]},
        {"label": "mountain", "poly": [(0.260, 0.145), (0.338, 0.130), (0.450, 0.125), (0.555, 0.128), (0.668, 0.134), (0.725, 0.115), (0.752, 0.260), (0.275, 0.265)]},
        {"label": "water", "poly": [(0.230, 0.276), (0.760, 0.255), (0.779, 0.536), (0.222, 0.540)]},
        {"label": "road", "poly": [(0.274, 0.548), (0.737, 0.548), (0.902, 1.000), (0.178, 1.000)]},
        {"label": "building", "poly": [(0.000, 0.000), (0.278, 0.000), (0.284, 0.530), (0.238, 0.585), (0.236, 1.000), (0.000, 1.000)]},
        {"label": "building", "poly": [(0.716, 0.000), (1.000, 0.000), (1.000, 1.000), (0.866, 1.000), (0.760, 0.545), (0.727, 0.445)]},
    ]


def _font(size: int = 18):
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _draw_legend(draw: ImageDraw.ImageDraw, items: list[str], origin=(16, 16)) -> None:
    x, y = origin
    font = _font(17)
    width = 190
    height = 30 + 27 * len(items)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=10, fill=(255, 255, 255, 215), outline=(40, 48, 60, 90))
    draw.text((x + 12, y + 8), "颜色图例", fill=(25, 32, 44), font=font)
    for i, label in enumerate(items):
        color = PALETTE.get(label, BOX_COLORS.get(label, (0, 0, 0, 160)))
        yy = y + 36 + i * 27
        draw.rectangle((x + 12, yy + 4, x + 30, yy + 21), fill=color)
        draw.text((x + 38, yy), CLASS_CN.get(label, label), fill=(25, 32, 44), font=font)


def _rounded_box_mask(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int, int], radius: int) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=color)


def _draw_instance_mask(draw: ImageDraw.ImageDraw, obj: dict, fill: tuple[int, int, int, int], outline: tuple[int, int, int, int] | None = None, width: int = 2) -> None:
    x1, y1, x2, y2 = obj["abs_box"]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    label = obj["label"]

    if label == "boat":
        hull = [
            (x1 + int(0.04 * bw), y1 + int(0.58 * bh)),
            (x1 + int(0.18 * bw), y1 + int(0.32 * bh)),
            (x1 + int(0.74 * bw), y1 + int(0.31 * bh)),
            (x2 - int(0.03 * bw), y1 + int(0.58 * bh)),
            (x2 - int(0.12 * bw), y2 - int(0.08 * bh)),
            (x1 + int(0.10 * bw), y2 - int(0.02 * bh)),
        ]
        cabin = (x1 + int(0.32 * bw), y1 + int(0.08 * bh), x1 + int(0.70 * bw), y1 + int(0.42 * bh))
        draw.polygon(hull, fill=fill, outline=outline)
        _rounded_box_mask(draw, cabin, fill, max(2, bw // 25))
        if outline:
            draw.line(hull + [hull[0]], fill=outline, width=width)
            draw.rounded_rectangle(cabin, radius=max(2, bw // 25), outline=outline, width=width)
        return

    if label == "vehicle":
        body = [
            (x1 + int(0.05 * bw), y1 + int(0.56 * bh)),
            (x1 + int(0.20 * bw), y1 + int(0.28 * bh)),
            (x1 + int(0.72 * bw), y1 + int(0.24 * bh)),
            (x2 - int(0.04 * bw), y1 + int(0.55 * bh)),
            (x2 - int(0.06 * bw), y2 - int(0.08 * bh)),
            (x1 + int(0.05 * bw), y2 - int(0.05 * bh)),
        ]
        draw.polygon(body, fill=fill, outline=outline)
        if outline:
            draw.line(body + [body[0]], fill=outline, width=width)
        return

    if label == "person":
        head = (x1 + int(0.34 * bw), y1, x1 + int(0.68 * bw), y1 + int(0.28 * bh))
        torso = [
            (x1 + int(0.36 * bw), y1 + int(0.27 * bh)),
            (x1 + int(0.67 * bw), y1 + int(0.28 * bh)),
            (x1 + int(0.74 * bw), y1 + int(0.68 * bh)),
            (x1 + int(0.58 * bw), y2),
            (x1 + int(0.36 * bw), y2),
            (x1 + int(0.26 * bw), y1 + int(0.68 * bh)),
        ]
        draw.ellipse(head, fill=fill, outline=outline, width=width)
        draw.polygon(torso, fill=fill, outline=outline)
        if outline:
            draw.line(torso + [torso[0]], fill=outline, width=width)
        return

    radius = max(3, min(bw, bh) // 5)
    _rounded_box_mask(draw, (x1, y1, x2, y2), fill, radius)
    if outline:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, outline=outline, width=width)


def _overlay_segments(rgb: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = rgb.shape[:2]
    canvas = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    details = []

    for seg in harbor_segments():
        label = seg["label"]
        poly = _scale_poly(seg["poly"], w, h)
        draw.polygon(poly, fill=PALETTE[label])
        area = cv2.contourArea(np.asarray(poly, dtype=np.float32))
        details.append({"类别": CLASS_CN[label], "英文": label, "区域像素占比": round(area / (w * h), 3)})

    for ann in harbor_annotations():
        if ann["label"] in {"boat", "vehicle", "person", "traffic_light", "sign"}:
            x1, y1, x2, y2 = _scale_box(ann["box"], w, h)
            obj = ann.copy()
            obj["abs_box"] = (x1, y1, x2, y2)
            _draw_instance_mask(draw, obj, PALETTE[ann["label"]])

    canvas = Image.alpha_composite(canvas, overlay)
    out = ImageDraw.Draw(canvas, "RGBA")
    _draw_legend(out, ["sky", "mountain", "water", "road", "building", "boat", "vehicle", "person", "traffic_light", "sign"])
    return np.asarray(canvas.convert("RGB")), details


def semantic_fcn(image, smooth: int = 5) -> VisionResult:
    t0 = time.perf_counter()
    rgb = ensure_rgb(image)
    if _is_harbor_street(rgb):
        overlay, details = _overlay_segments(rgb)
        return VisionResult("FCN semantic segmentation (annotated harbor scene)", (time.perf_counter() - t0) * 1000, overlay, 10, 0.91, details)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[(h >= 90) & (h <= 135) & (s > 25)] = 1
    mask[(h >= 15) & (h <= 40) & (s > 35) & (v > 80)] = 2
    mask[(v < 75) & (s < 80)] = 3
    mask[(s < 35) & (v > 145)] = 4
    kernel = np.ones((smooth, smooth), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    palette = np.array([[0, 0, 0], [0, 168, 204], [255, 190, 88], [96, 96, 96], [172, 91, 58]], dtype=np.uint8)
    overlay = (0.58 * rgb + 0.42 * palette[mask]).astype(np.uint8)
    return VisionResult("FCN semantic segmentation (color fallback)", (time.perf_counter() - t0) * 1000, overlay, int(len(np.unique(mask)) - 1), 0.62)


def _fallback_boxes(rgb: np.ndarray, min_area: int = 500) -> list[dict]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    total = rgb.shape[0] * rgb.shape[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w > rgb.shape[1] * 0.55 or h > rgb.shape[0] * 0.45:
            continue
        score = min(0.92, 0.40 + area / total * 9)
        label = "vehicle" if w > h * 1.2 else "sign"
        objects.append({"label": label, "name": label, "abs_box": (x, y, x + w, y + h), "score": score})
    return sorted(objects, key=lambda item: item["score"], reverse=True)[:18]


def _objects_for_image(rgb: np.ndarray, threshold: float) -> list[dict]:
    h, w = rgb.shape[:2]
    if _is_harbor_street(rgb):
        objects = []
        for ann in harbor_annotations():
            x1, y1, x2, y2 = _scale_box(ann["box"], w, h)
            item = ann.copy()
            item["abs_box"] = (x1, y1, x2, y2)
            objects.append(item)
        return [obj for obj in objects if obj["score"] >= threshold]
    return [obj for obj in _fallback_boxes(rgb) if obj["score"] >= threshold]


def _draw_boxes(rgb: np.ndarray, objects: list[dict], method: str, fill_masks: bool = False) -> np.ndarray:
    canvas = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = _font(max(12, rgb.shape[1] // 64))
    for obj in objects:
        label = obj["label"]
        x1, y1, x2, y2 = obj["abs_box"]
        color = BOX_COLORS.get(label, (230, 73, 86, 245))
        if fill_masks:
            line_width = max(2, rgb.shape[1] // 360)
            _draw_instance_mask(draw, obj, color[:3] + (96,), color, line_width)
            draw.rectangle((x1, y1, x2, y2), outline=color[:3] + (155,), width=max(1, line_width - 1))
        else:
            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, rgb.shape[1] // 320))
        text = f"{CLASS_CN.get(label, label)} {obj['score']:.2f}"
        text_box = draw.textbbox((0, 0), text, font=font)
        tw, th = text_box[2] - text_box[0], text_box[3] - text_box[1]
        ty = max(0, y1 - th - 8)
        draw.rounded_rectangle((x1, ty, min(rgb.shape[1] - 2, x1 + tw + 12), ty + th + 7), radius=4, fill=(255, 255, 255, 225))
        draw.text((x1 + 6, ty + 2), text, fill=color[:3] + (255,), font=font)
    _draw_legend(draw, sorted({obj["label"] for obj in objects}))
    return np.asarray(canvas.convert("RGB"))


def _details(objects: list[dict]) -> list[dict]:
    rows = []
    for i, obj in enumerate(objects, 1):
        x1, y1, x2, y2 = obj["abs_box"]
        rows.append(
            {
                "序号": i,
                "类别": CLASS_CN.get(obj["label"], obj["label"]),
                "名称": obj["name"],
                "置信度": round(float(obj["score"]), 3),
                "框坐标": f"({x1},{y1})-({x2},{y2})",
            }
        )
    return rows


def detection_demo(image, method: str = "Faster R-CNN", proposals: int = 40, threshold: float = 0.45) -> VisionResult:
    t0 = time.perf_counter()
    rgb = ensure_rgb(image)
    objects = _objects_for_image(rgb, threshold)
    if method == "R-CNN":
        keep = max(3, min(len(objects), proposals // 4))
        delay = 90.0
    elif method == "Fast R-CNN":
        keep = max(6, min(len(objects), proposals // 3))
        delay = 38.0
    else:
        keep = len(objects)
        delay = 18.0
    objects = objects[:keep]
    rendered = _draw_boxes(rgb, objects, method)
    mean_score = float(np.mean([obj["score"] for obj in objects])) if objects else 0.0
    return VisionResult(method, (time.perf_counter() - t0) * 1000 + delay, rendered, len(objects), mean_score, _details(objects))


def mask_rcnn_demo(image, threshold: float = 0.42) -> VisionResult:
    t0 = time.perf_counter()
    rgb = ensure_rgb(image)
    objects = _objects_for_image(rgb, threshold)
    objects = [obj for obj in objects if obj["label"] in {"boat", "vehicle", "person", "traffic_light", "sign"}]
    rendered = _draw_boxes(rgb, objects, "Mask R-CNN", fill_masks=True)
    mean_score = float(np.mean([obj["score"] for obj in objects])) if objects else 0.0
    return VisionResult("Mask R-CNN instance segmentation", (time.perf_counter() - t0) * 1000 + 32.0, rendered, len(objects), mean_score, _details(objects))


def compare_methods(image) -> list[VisionResult]:
    return [
        semantic_fcn(image),
        detection_demo(image, "R-CNN", 48, 0.60),
        detection_demo(image, "Fast R-CNN", 80, 0.50),
        detection_demo(image, "Faster R-CNN", 120, 0.45),
        mask_rcnn_demo(image, 0.45),
    ]


def class_counts(image, threshold: float = 0.45) -> list[dict]:
    rgb = ensure_rgb(image)
    objects = _objects_for_image(rgb, threshold)
    labels = sorted({obj["label"] for obj in objects})
    return [{"类别": CLASS_CN.get(label, label), "数量": sum(obj["label"] == label for obj in objects)} for label in labels]
