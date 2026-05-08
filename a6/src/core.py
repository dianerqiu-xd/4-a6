from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass
class VisionResult:
    method: str
    latency_ms: float
    image: np.ndarray
    count: int
    score: float


def ensure_rgb(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3].astype(np.uint8)


def default_scene(size: tuple[int, int] = (360, 260)) -> np.ndarray:
    w, h = size
    img = Image.new("RGB", (w, h), (235, 239, 244))
    d = ImageDraw.Draw(img)
    d.rectangle((30, 150, 150, 225), fill=(55, 128, 224))
    d.ellipse((210, 50, 310, 150), fill=(225, 74, 92))
    d.polygon([(180, 210), (240, 120), (310, 220)], fill=(41, 170, 118))
    d.line((0, 236, w, 236), fill=(105, 117, 135), width=5)
    return np.asarray(img)


def semantic_fcn(image, smooth: int = 5) -> VisionResult:
    t0 = time.perf_counter()
    rgb = ensure_rgb(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[(s < 35) & (v > 150)] = 1
    mask[(h < 12) | (h > 168)] = 2
    mask[(h >= 36) & (h <= 88) & (s > 40)] = 3
    mask[(h >= 90) & (h <= 130) & (s > 40)] = 4
    kernel = np.ones((smooth, smooth), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    palette = np.array([[0, 0, 0], [160, 160, 160], [230, 73, 86], [48, 180, 112], [55, 132, 229]], dtype=np.uint8)
    overlay = (0.58 * rgb + 0.42 * palette[mask]).astype(np.uint8)
    return VisionResult("FCN semantic segmentation", (time.perf_counter() - t0) * 1000, overlay, int(len(np.unique(mask)) - 1), float(mask.mean()))


def _contour_boxes(rgb: np.ndarray, min_area: int = 350) -> list[tuple[int, int, int, int, float]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    total = rgb.shape[0] * rgb.shape[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        score = min(0.98, 0.35 + area / total * 7.0)
        boxes.append((x, y, x + w, y + h, score))
    return sorted(boxes, key=lambda b: b[4], reverse=True)


def detection_demo(image, method: str = "Faster R-CNN", proposals: int = 40, threshold: float = 0.45) -> VisionResult:
    t0 = time.perf_counter()
    rgb = ensure_rgb(image)
    boxes = _contour_boxes(rgb)
    if method == "R-CNN":
        boxes = boxes[:max(1, proposals // 8)]
        delay = 4.0
    elif method == "Fast R-CNN":
        boxes = boxes[:max(1, proposals // 5)]
        delay = 2.2
    else:
        boxes = boxes[:max(1, proposals // 10)]
        delay = 1.0
    boxes = [b for b in boxes if b[4] >= threshold]
    canvas = Image.fromarray(rgb)
    d = ImageDraw.Draw(canvas, "RGBA")
    for i, (x1, y1, x2, y2, score) in enumerate(boxes):
        color = [(230, 73, 86, 230), (55, 132, 229, 230), (43, 171, 118, 230)][i % 3]
        d.rectangle((x1, y1, x2, y2), outline=color, width=4)
        d.rectangle((x1, max(0, y1 - 22), x1 + 128, y1), fill=(255, 255, 255, 210))
        d.text((x1 + 4, max(0, y1 - 19)), f"object {score:.2f}", fill=color)
    latency = (time.perf_counter() - t0) * 1000 + delay
    mean_score = float(np.mean([b[4] for b in boxes])) if boxes else 0.0
    return VisionResult(method, latency, np.asarray(canvas), len(boxes), mean_score)


def mask_rcnn_demo(image, threshold: float = 0.42) -> VisionResult:
    t0 = time.perf_counter()
    rgb = ensure_rgb(image)
    boxes = [b for b in _contour_boxes(rgb) if b[4] >= threshold]
    canvas = Image.fromarray(rgb)
    d = ImageDraw.Draw(canvas, "RGBA")
    for i, (x1, y1, x2, y2, score) in enumerate(boxes):
        color = [(230, 73, 86, 90), (55, 132, 229, 90), (43, 171, 118, 90)][i % 3]
        d.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=color, outline=color[:3] + (235,), width=3)
    return VisionResult("Mask R-CNN instance segmentation", (time.perf_counter() - t0) * 1000 + 3.4, np.asarray(canvas), len(boxes), float(np.mean([b[4] for b in boxes])) if boxes else 0.0)


def compare_methods(image) -> list[VisionResult]:
    return [
        semantic_fcn(image),
        detection_demo(image, "R-CNN", 48, 0.35),
        detection_demo(image, "Fast R-CNN", 80, 0.35),
        detection_demo(image, "Faster R-CNN", 60, 0.35),
        mask_rcnn_demo(image, 0.35),
    ]
