from __future__ import annotations

import base64, io, json, time, uuid, datetime as dt
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO
import onnxruntime as ort

try:
    from streamlit.elements.image import image_to_url as _tmp  # noqa: F401
except ImportError:  # Streamlit bản mới đã bỏ hàm này
    import streamlit.elements.image as _st_img
    import base64, io
    from PIL import Image

    def _image_to_url(img, *args, kind: str = "png", **kwargs) -> str:  # 👈 chấp nhận mọi arg
        """
        Fallback for streamlit_drawable_canvas.
        Những tham số ngoài `img` bị bỏ qua vì ta chỉ cần trả về data-URI.
        """
        if isinstance(img, Image.Image):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{b64}"
        if isinstance(img, str):            # URL hoặc path file
            return img
        raise TypeError("Unsupported image type")

    _st_img.image_to_url = _image_to_url  # type: ignore
# -----------------------------------------------------------------

# -----------------------------------------------------------------------------
LABEL_ORDER = [
    "dish_empty", "dish_kakigori", "dish_not_empty",
    "tray_empty", "tray_kakigori", "tray_not_empty",
]
CLS_DISH_KAKI, CLS_TRAY_KAKI = 1, 4
RECLASS_EVERY_DEFAULT = 30

MODEL_DETECT = "./models/detection_model.pt"
MODEL_CLASS  = "./models/classification_model.onnx"
DEEPSORT_KW  = dict(max_age=30)

TMP_DIR = Path("tmp_upload"); TMP_DIR.mkdir(exist_ok=True)
FEED_DIR = Path("feedback");  FEED_DIR.mkdir(exist_ok=True)
FEED_IMG_DIR  = FEED_DIR / "images";     FEED_IMG_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def np2pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def save_feedback(
    tid: int,
    pred_idx: int,
    correct_idx: int,
    crop_img: Image.Image,
) -> None:
    """Lưu ảnh crop + metadata"""
    ts   = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    name = f"{ts}_id{tid}_pred{pred_idx}_corr{correct_idx}.png"
    img_path = FEED_IMG_DIR / name
    crop_img.save(img_path, format="PNG")

    rec = {
        "ts": ts,
        "track_id": tid,
        "pred": pred_idx,
        "correct": correct_idx,
        "img": str(img_path.relative_to(FEED_DIR)),
    }
    log_file = FEED_DIR / f"fb_{dt.date.today():%Y%m%d}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def load_runtime(device: str):
    yolo = YOLO(MODEL_DETECT).to(device)
    tracker = DeepSort(**DEEPSORT_KW)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(MODEL_CLASS, providers=providers)
    prep = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return yolo, tracker, sess, prep


def select_roi(first_bgr, max_w=720):
    import streamlit as st
    from streamlit_drawable_canvas import st_canvas
    from PIL import Image
    import cv2, numpy as np

    # ─── Khởi tạo session state ─────────────────────────────────
    st.session_state.setdefault("roi_tmp", None)
    st.session_state.setdefault("roi_ok", False)

    # ─── Giảm kích thước hiển thị ──────────────────────────────
    h_org, w_org = first_bgr.shape[:2]
    scale = min(1.0, max_w / w_org)
    disp = cv2.resize(first_bgr, (int(w_org*scale), int(h_org*scale)))
    disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    img_pil  = Image.fromarray(disp_rgb)

    # ─── 4 placeholder toạ độ luôn ở đầu trang ─────────────────
    col = st.columns(4)
    ph_x1 = col[0].empty(); ph_y1 = col[1].empty()
    ph_x2 = col[2].empty(); ph_y2 = col[3].empty()

    # ─── Canvas cho người dùng kéo ROI ─────────────────────────
    res = st_canvas(
        drawing_mode="rect",
        fill_color="rgba(0,255,0,0.2)",
        stroke_width=2,
        background_image=img_pil,
        width=disp.shape[1],
        height=disp.shape[0],
        key="roi_canvas",
        update_streamlit=True,
    )

    # ─── Nếu có hình chữ nhật, lấy toạ độ & hiển thị ───────────
    if res.json_data and res.json_data["objects"]:
        obj = res.json_data["objects"][-1]
        x1_s, y1_s = obj["left"], obj["top"]
        x2_s, y2_s = x1_s + obj["width"], y1_s + obj["height"]

        inv = 1 / scale
        x1, y1, x2, y2 = map(int, (x1_s*inv, y1_s*inv, x2_s*inv, y2_s*inv))
        st.session_state["roi_tmp"] = (x1, y1, x2, y2)

        # 🔄 cập-nhật placeholder (không đụng session_state của widget)
        ph_x1.number_input(label="x1", value=x1, disabled=True, label_visibility="collapsed", key="disp_x1")
        ph_y1.number_input(label="y1", value=y1, disabled=True, label_visibility="collapsed", key="disp_y1")
        ph_x2.number_input(label="x2", value=x2, disabled=True, label_visibility="collapsed", key="disp_x2")
        ph_y2.number_input(label="y2", value=y2, disabled=True, label_visibility="collapsed", key="disp_y2")
    else:
        # hiển thị rỗng
        for idx, ph in enumerate((ph_x1, ph_y1, ph_x2, ph_y2)):
            ph.number_input(
                label="dummy",               # nhãn trống
                value=0,
                disabled=True,
                label_visibility="collapsed",
                key=f"dummy_{idx}"      # 👈 thêm key duy nhất
            )

    # ─── Confirm / Reset ───────────────────────────────────────
    roi_ph = st.empty()
    with roi_ph.container():
        cf, rs = st.columns(2)
        if cf.button("✅ Confirm ROI", disabled=st.session_state["roi_ok"]):
            if st.session_state["roi_tmp"]:
                st.session_state["roi_ok"] = True
                st.success("ROI confirmed!")
                roi_ph.empty()  # xoá placeholder
                st.rerun() 
            else:
                st.warning("Bạn chưa vẽ ROI.")
        if st.button("🔄 Reset ROI"):
                st["roi_tmp"], st["roi_ok"] = None, False
                st.rerun()
            

    return st.session_state["roi_tmp"] if st.session_state["roi_ok"] and st.session_state["roi_tmp"] else None

# -----------------------------------------------------------------------------
# Streamlit main
# -----------------------------------------------------------------------------

def main():
    st.set_page_config("Kakigori Counter", layout="wide")
    st.title("🍧 Kakigori Dish/Tray Monitoring System")

    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if uploaded is None:
        st.stop()
    tmp_path = TMP_DIR / f"{uuid.uuid4()}_{uploaded.name}"
    with open(tmp_path, "wb") as f: f.write(uploaded.read())

    cap = cv2.VideoCapture(str(tmp_path)); ok, first = cap.read()
    if not ok or first is None:
        st.error("Cannot read video"); st.stop()

    roi = select_roi(first)
    if roi:
        x1, y1, x2, y2 = roi
        st.success(f"ROI: ({x1},{y1}) – ({x2},{y2})")
    else:
        st.stop()

    re_every = st.number_input(label="Re‑classify every N frames", min_value=10, max_value=120, value=RECLASS_EVERY_DEFAULT,disabled=True,label_visibility="collapsed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner("Loading models …"):
        yolo, tracker, sess, prep = load_runtime(device)
    st.write("Running on", device.upper())

    vid_box = st.empty(); cnt_box = st.empty()

    track_labels: Dict[int, int] = {}
    last_pred: Dict[int, int]   = {}
    dish_ids, tray_ids = set(), set()
    dish_cnt = tray_cnt = 0

    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        sub = frame[y1:y2, x1:x2]

        res = yolo.predict(sub, imgsz=640, conf=0.4, verbose=False)[0]
        dets = [[
            [b[0], b[1], b[2]-b[0], b[3]-b[1]], float(c), int(cl)]
            for b, c, cl in zip(res.boxes.xyxy.cpu().numpy(),
                                 res.boxes.conf.cpu().numpy(),
                                 res.boxes.cls.cpu().numpy())]
        tracks = tracker.update_tracks(dets, frame=sub)
        current_ids = set()

        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            current_ids.add(tid)
            need = tid not in last_pred or frame_idx - last_pred[tid] >= re_every
            if need:
                bx1, by1, bx2, by2 = map(int, t.to_ltrb())
                crop = sub[by1:by2, bx1:bx2]
                if crop.size:
                    img = np2pil(crop)
                    inp = prep(img).unsqueeze(0).numpy()
                    cls_pred = int(sess.run(None, {"input": inp})[0].argmax())
                    # track_labels[tid] = cls_pred
                    last_pred[tid] = frame_idx
                    
                    change_cnt = st.session_state.setdefault("change_cnt", {})  # id -> count
                    
                    changed = tid not in track_labels or cls_pred != track_labels[tid]
                    if changed:
                        track_labels[tid] = cls_pred   # cập-nhật nhãn “đang dùng”
                        change_cnt[tid]  = change_cnt.get(tid, 0) + 1

                        # ─── ❷  Hiển thị hộp feedback duy nhất cho (tid, cls_pred) ──
                        uid = int(time.time() * 1000)  # unique ID cho mỗi feedback
                        fb_prefix = f"fb_{tid}_{cls_pred}_uid"     # key duy nhất cho combo này
                        sel_key   = fb_prefix + "_sel"
                        but_key   = fb_prefix + "_but"
                        saved_key = fb_prefix + "_saved"

                        if saved_key not in st.session_state:         # CHƯA feedback cho (id, class)
                            ph = st.empty()                           # placeholder gói toàn bộ expander
                            st.warning("🛑 Video paused – please give feedback")
                            with ph.expander(f"Feedback for ID {tid}", expanded=True):
                                st.image(img, caption=f"Pred: {LABEL_ORDER[cls_pred]}")

                                correct_lbl = st.selectbox(
                                    "Correct label?", LABEL_ORDER,
                                    index=cls_pred, key=sel_key
                                )

                                if st.button("Save feedback", key=but_key):
                                    save_feedback(
                                        tid, cls_pred, LABEL_ORDER.index(correct_lbl), img
                                    )
                                    st.session_state[saved_key] = True
                                    ph.empty()                        # xoá cả expander & header
                                    st.rerun()
                            st.stop()
                    
                    if cls_pred == CLS_DISH_KAKI and tid not in dish_ids:
                        dish_ids.add(tid); dish_cnt += 1
                    elif cls_pred == CLS_TRAY_KAKI and tid not in tray_ids:
                        tray_ids.add(tid); tray_cnt += 1

            # draw bbox
            if tid in track_labels:
                cls_id = track_labels[tid]
                bx1, by1, bx2, by2 = map(int, t.to_ltrb())
                cv2.rectangle(frame, (x1+bx1, y1+by1), (x1+bx2, y1+by2), (0,255,0), 2)
                cv2.putText(frame, f"{tid}:{LABEL_ORDER[cls_id]}", (x1+bx1, y1+by1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # remove vanished ids from present sets only (counts persist)
        vanished = set(track_labels) - current_ids
        for vid in vanished:
            dish_ids.discard(vid); tray_ids.discard(vid)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 1)
        cv2.putText(frame, f"dish_kakigori: {dish_cnt}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame, f"tray_kakigori: {tray_cnt}", (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        
        vid_box.image(np2pil(frame), channels="RGB")

    st.success("Video finished")
    st.write({"dish_kakigori": dish_cnt, "tray_kakigori": tray_cnt})


if __name__ == "__main__":
    main()
