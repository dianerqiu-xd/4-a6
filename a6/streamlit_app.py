from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import core

st.set_page_config(page_title="A6 语义分割与目标检测 Vibe Coding", layout="wide")


def load_default_image() -> np.ndarray:
    for name in ["default_image.jpg", "default_image.jpeg", "default_image.png"]:
        path = ROOT / "assets" / name
        if path.exists():
            return np.asarray(Image.open(path).convert("RGB"))
    return np.zeros((240, 360, 3), dtype=np.uint8) + 235


def show_array(arr, caption: str):
    st.image(np.asarray(arr), caption=caption, use_container_width=True)


st.title("A6 语义分割与目标检测 Vibe Coding")
st.caption("学生：裘典儿 2025213456 · Agent/LLM：Codex GPT-5")

uploaded = st.sidebar.file_uploader("上传图片替换默认素材", type=["jpg", "jpeg", "png"])
if uploaded:
    image = np.asarray(Image.open(uploaded).convert("RGB"))
    image_source = f"上传图片：{uploaded.name}"
else:
    image = load_default_image()
    image_source = "默认图片：assets/default_image.jpg"

st.sidebar.image(image, caption=image_source, use_container_width=True)

tab_names = ['FCN', 'R-CNN系列', 'Mask R-CNN', '性能对比', '部署']
tabs = st.tabs(tab_names)


with tabs[0]:
    st.subheader("FCN 语义分割")
    st.image(image, caption=image_source, use_container_width=True)
    smooth = st.slider("平滑核大小", 3, 11, 5, 2)
    res = core.semantic_fcn(image, smooth=smooth)
    show_array(res.image, f"{res.method} · {res.latency_ms:.1f} ms")
    st.metric("语义区域数", res.count)

with tabs[1]:
    st.subheader("R-CNN / Fast / Faster R-CNN")
    method = st.selectbox("检测方法", ["R-CNN", "Fast R-CNN", "Faster R-CNN"])
    threshold = st.slider("置信度阈值", 0.2, 0.9, 0.42, 0.02)
    proposals = st.slider("候选框数量", 10, 120, 60, 5)
    res = core.detection_demo(image, method, proposals, threshold)
    show_array(res.image, f"{res.method} · {res.latency_ms:.1f} ms")
    st.metric("检测数量", res.count)

with tabs[2]:
    st.subheader("Mask R-CNN 实例分割")
    threshold = st.slider("Mask 阈值", 0.2, 0.9, 0.42, 0.02)
    res = core.mask_rcnn_demo(image, threshold)
    show_array(res.image, f"{res.method} · {res.latency_ms:.1f} ms")
    st.metric("实例数量", res.count)

with tabs[3]:
    st.subheader("方法性能对比")
    rows = [r.__dict__ for r in core.compare_methods(image)]
    st.dataframe([{k: v for k, v in row.items() if k != "image"} for row in rows], use_container_width=True)
    st.bar_chart({row["method"]: row["latency_ms"] for row in rows})


with tabs[-1]:
    st.subheader("部署说明")
    st.write("上传本文件夹到 GitHub 后，如果把该文件夹作为仓库根目录，Streamlit Cloud 的 Main file path 填 `streamlit_app.py`。")
    st.write("如果上传整个周四作业目录，则 Main file path 填 `a6/streamlit_app.py`。")
    st.write("本应用保留上传控件，可用自己的图片替换默认素材。")
