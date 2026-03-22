import math
import queue
import threading
import time
from collections import deque
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import serial
import serial.tools.list_ports
import streamlit as st


# =========================
# Config
# =========================
st.set_page_config(page_title="Realtime Magnetic Tactile Inference", layout="wide")


# =========================
# Utilities
# =========================
def list_serial_ports():
    return [p.device for p in serial.tools.list_ports.comports()]


def parse_line(line: str):
    """
    Supported formats:
      gx,gy,gz,B
      gx,gy,gz
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) not in (3, 4):
        return None

    try:
        gx = float(parts[0])
        gy = float(parts[1])
        gz = float(parts[2])
        if len(parts) == 4:
            B = float(parts[3])
        else:
            B = math.sqrt(gx * gx + gy * gy + gz * gz)

        return {
            "timestamp": time.time(),
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "B": B,
            "raw": line,
        }
    except ValueError:
        return None


def serial_reader_worker(port, baudrate, output_queue, stop_event):
    ser = None
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
        time.sleep(2.0)  # Arduino reset after serial open

        while not stop_event.is_set():
            try:
                raw = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue
                parsed = parse_line(raw)
                if parsed is not None:
                    output_queue.put(parsed)
            except Exception as e:
                output_queue.put({"error": f"Read error: {e}"})
                time.sleep(0.2)
    except Exception as e:
        output_queue.put({"error": f"Serial open error: {e}"})
    finally:
        if ser and ser.is_open:
            ser.close()


def safe_rise_time(signal: np.ndarray, time_arr: np.ndarray) -> float:
    if len(signal) < 3:
        return np.nan

    s0 = np.median(signal[: max(1, len(signal) // 10)])
    smax = np.max(signal)
    amp = smax - s0
    if abs(amp) < 1e-9:
        return np.nan

    low = s0 + 0.1 * amp
    high = s0 + 0.9 * amp

    try:
        t_low = time_arr[np.where(signal >= low)[0][0]]
        t_high = time_arr[np.where(signal >= high)[0][0]]
        return float(t_high - t_low)
    except Exception:
        return np.nan


def zero_crossing_rate(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    signs = np.sign(x)
    return float(np.mean(signs[1:] != signs[:-1]))


def extract_features_from_window(df_window: pd.DataFrame) -> pd.DataFrame:
    """
    Must match training-time feature logic as closely as possible.
    Returns one-row DataFrame.
    """
    seg = df_window.sort_values("timestamp").reset_index(drop=True)

    t = seg["timestamp"].to_numpy(dtype=float)
    t_rel = t - t[0]

    gx = seg["gx"].to_numpy(dtype=float)
    gy = seg["gy"].to_numpy(dtype=float)
    gz = seg["gz"].to_numpy(dtype=float)
    B = seg["B"].to_numpy(dtype=float)

    n = len(seg)
    base_n = max(3, int(0.2 * n))
    base_slice = slice(0, base_n)

    gx0 = float(np.mean(gx[base_slice]))
    gy0 = float(np.mean(gy[base_slice]))
    gz0 = float(np.mean(gz[base_slice]))
    B0 = float(np.mean(B[base_slice]))

    dgx = gx - gx0
    dgy = gy - gy0
    dgz = gz - gz0
    dB = B - B0

    dt = np.diff(t_rel)
    dt[dt == 0] = 1e-6

    dB_dt = np.diff(B) / dt if len(B) > 1 else np.array([0.0])
    dgx_dt = np.diff(gx) / dt if len(gx) > 1 else np.array([0.0])
    dgy_dt = np.diff(gy) / dt if len(gy) > 1 else np.array([0.0])
    dgz_dt = np.diff(gz) / dt if len(gz) > 1 else np.array([0.0])

    start_hold = int(0.3 * n)
    end_hold = max(start_hold + 1, int(0.7 * n))
    hold_B = B[start_hold:end_hold]
    hold_dB = dB[start_hold:end_hold]

    features = {
        "n_samples": int(n),
        "duration_sec": float(t_rel[-1]) if n > 1 else 0.0,

        "gx_mean": float(np.mean(gx)),
        "gy_mean": float(np.mean(gy)),
        "gz_mean": float(np.mean(gz)),
        "B_mean": float(np.mean(B)),

        "gx_std": float(np.std(gx)),
        "gy_std": float(np.std(gy)),
        "gz_std": float(np.std(gz)),
        "B_std": float(np.std(B)),

        "dgx_mean": float(np.mean(dgx)),
        "dgy_mean": float(np.mean(dgy)),
        "dgz_mean": float(np.mean(dgz)),
        "dB_mean": float(np.mean(dB)),

        "dgx_max_abs": float(np.max(np.abs(dgx))),
        "dgy_max_abs": float(np.max(np.abs(dgy))),
        "dgz_max_abs": float(np.max(np.abs(dgz))),
        "dB_max_abs": float(np.max(np.abs(dB))),

        "gx_range": float(np.max(gx) - np.min(gx)),
        "gy_range": float(np.max(gy) - np.min(gy)),
        "gz_range": float(np.max(gz) - np.min(gz)),
        "B_range": float(np.max(B) - np.min(B)),

        "dB_dt_max": float(np.max(dB_dt)) if len(dB_dt) else 0.0,
        "dB_dt_min": float(np.min(dB_dt)) if len(dB_dt) else 0.0,
        "dB_dt_max_abs": float(np.max(np.abs(dB_dt))) if len(dB_dt) else 0.0,

        "dgx_dt_max_abs": float(np.max(np.abs(dgx_dt))) if len(dgx_dt) else 0.0,
        "dgy_dt_max_abs": float(np.max(np.abs(dgy_dt))) if len(dgy_dt) else 0.0,
        "dgz_dt_max_abs": float(np.max(np.abs(dgz_dt))) if len(dgz_dt) else 0.0,

        "rise_time_B": safe_rise_time(B, t_rel),
        "rise_time_dB": safe_rise_time(dB, t_rel),

        "hold_B_mean": float(np.mean(hold_B)) if len(hold_B) else np.nan,
        "hold_B_std": float(np.std(hold_B)) if len(hold_B) else np.nan,
        "hold_dB_mean": float(np.mean(hold_dB)) if len(hold_dB) else np.nan,
        "hold_dB_std": float(np.std(hold_dB)) if len(hold_dB) else np.nan,

        "B_peak_idx_ratio": float(np.argmax(B) / max(1, n - 1)),
        "dB_peak_idx_ratio": float(np.argmax(np.abs(dB)) / max(1, n - 1)),
        "B_zero_cross_rate": float(zero_crossing_rate(dB)),
        "B_energy": float(np.mean(B ** 2)),
        "dB_energy": float(np.mean(dB ** 2)),

        "gx_last": float(gx[-1]),
        "gy_last": float(gy[-1]),
        "gz_last": float(gz[-1]),
        "B_last": float(B[-1]),
        "dB_last": float(dB[-1]),
    }

    return pd.DataFrame([features])


def predict_three_class(df_window: pd.DataFrame, model_contact, model_tp):
    feat_df = extract_features_from_window(df_window)

    pred_contact = model_contact.predict(feat_df)[0]

    prob_contact = None
    if hasattr(model_contact, "predict_proba"):
        p = model_contact.predict_proba(feat_df)[0]
        prob_contact = dict(zip(model_contact.classes_, p))

    if pred_contact == "no_contact":
        result = {
            "pred_3class": "no_contact",
            "pred_contact": pred_contact,
            "pred_touch_punch": None,
            "prob_contact": prob_contact,
            "prob_touch_punch": None,
            "features": feat_df,
        }
        return result

    pred_tp = model_tp.predict(feat_df)[0]

    prob_tp = None
    if hasattr(model_tp, "predict_proba"):
        p2 = model_tp.predict_proba(feat_df)[0]
        prob_tp = dict(zip(model_tp.classes_, p2))

    result = {
        "pred_3class": pred_tp,
        "pred_contact": pred_contact,
        "pred_touch_punch": pred_tp,
        "prob_contact": prob_contact,
        "prob_touch_punch": prob_tp,
        "features": feat_df,
    }
    return result


def make_plot(df_plot: pd.DataFrame, channels):
    fig = go.Figure()
    for ch in channels:
        fig.add_trace(
            go.Scatter(
                x=df_plot["t"],
                y=df_plot[ch],
                mode="lines",
                name=ch
            )
        )
    fig.update_layout(
        height=450,
        xaxis_title="Time (s)",
        yaxis_title="Magnetic reading",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


# =========================
# Session State Init
# =========================
if "running" not in st.session_state:
    st.session_state.running = False
if "reader_thread" not in st.session_state:
    st.session_state.reader_thread = None
if "stop_event" not in st.session_state:
    st.session_state.stop_event = None
if "data_queue" not in st.session_state:
    st.session_state.data_queue = queue.Queue()
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=5000)
if "messages" not in st.session_state:
    st.session_state.messages = deque(maxlen=20)
if "pred_history" not in st.session_state:
    st.session_state.pred_history = deque(maxlen=50)
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# =========================
# Sidebar
# =========================
st.title("Realtime Magnetic Tactile Detection")

with st.sidebar:
    st.header("Connection")
    ports = list_serial_ports()
    port = st.selectbox("Serial Port", ports if ports else [""])
    baudrate = st.selectbox("Baud Rate", [9600, 19200, 38400, 57600, 115200], index=4)

    st.header("Model Paths")
    model_contact_path = st.text_input(
        "contact vs no_contact model",
        value=os.environ.get("MODEL_CONTACT_PATH", "outputs/contact_vs_no_contact_logreg.joblib")
    )
    model_tp_path = st.text_input(
        "touch vs punch model",
        value=os.environ.get("MODEL_TP_PATH", "outputs/touch_vs_punch_logreg.joblib")
    )

    st.header("Realtime Inference")
    window_sec = st.slider("Window Length (sec)", 0.5, 3.0, 1.5, 0.1)
    min_samples = st.slider("Min Samples for Prediction", 10, 200, 30, 5)
    refresh_sec = st.slider("Refresh Interval (sec)", 0.1, 2.0, 0.3, 0.1)
    visible_points = st.slider("Visible Plot Points", 100, 2000, 400, 50)

    st.header("Channels")
    show_gx = st.checkbox("gx", True)
    show_gy = st.checkbox("gy", True)
    show_gz = st.checkbox("gz", True)
    show_B = st.checkbox("B", True)

    col1, col2 = st.columns(2)
    connect_clicked = col1.button("Connect", use_container_width=True)
    disconnect_clicked = col2.button("Disconnect", use_container_width=True)

    clear_hist = st.button("Clear Prediction History", use_container_width=True)

    if clear_hist:
        st.session_state.pred_history.clear()
        st.session_state.messages.appendleft("Prediction history cleared.")

# =========================
# Load Models
# =========================
model_contact = None
model_tp = None
model_load_error = None

try:
    model_contact = joblib.load(model_contact_path)
    model_tp = joblib.load(model_tp_path)
except Exception as e:
    model_load_error = str(e)

# =========================
# Connection Management
# =========================
if connect_clicked and not st.session_state.running:
    st.session_state.buffer = deque(maxlen=5000)
    st.session_state.messages.clear()
    st.session_state.stop_event = threading.Event()
    st.session_state.data_queue = queue.Queue()

    st.session_state.reader_thread = threading.Thread(
        target=serial_reader_worker,
        args=(port, baudrate, st.session_state.data_queue, st.session_state.stop_event),
        daemon=True,
    )
    st.session_state.reader_thread.start()
    st.session_state.running = True
    st.session_state.messages.appendleft(f"Connected to {port} @ {baudrate}")

if disconnect_clicked and st.session_state.running:
    st.session_state.stop_event.set()
    st.session_state.running = False
    st.session_state.messages.appendleft("Serial disconnected.")

# =========================
# Drain Queue
# =========================
while not st.session_state.data_queue.empty():
    item = st.session_state.data_queue.get()
    if "error" in item:
        st.session_state.messages.appendleft(item["error"])
        st.session_state.running = False
    else:
        st.session_state.buffer.append(item)

df = pd.DataFrame(list(st.session_state.buffer))
if not df.empty:
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    df_plot = df.tail(visible_points).copy()
else:
    df_plot = pd.DataFrame(columns=["t", "gx", "gy", "gz", "B"])

# =========================
# Realtime Prediction
# =========================
prediction_result = None

if model_load_error:
    st.error(f"Model load error: {model_load_error}")
elif not df.empty:
    latest_time = df["timestamp"].iloc[-1]
    df_window = df[df["timestamp"] >= latest_time - window_sec].copy()

    if len(df_window) >= min_samples:
        try:
            prediction_result = predict_three_class(df_window, model_contact, model_tp)
            st.session_state.last_prediction = prediction_result

            history_item = {
                "timestamp": latest_time,
                "pred_3class": prediction_result["pred_3class"],
            }

            if prediction_result["prob_contact"] is not None:
                for k, v in prediction_result["prob_contact"].items():
                    history_item[f"contact_prob_{k}"] = v

            if prediction_result["prob_touch_punch"] is not None:
                for k, v in prediction_result["prob_touch_punch"].items():
                    history_item[f"tp_prob_{k}"] = v

            # avoid flooding identical timestamps
            if not st.session_state.pred_history or latest_time != st.session_state.pred_history[-1]["timestamp"]:
                st.session_state.pred_history.append(history_item)

        except Exception as e:
            st.session_state.messages.appendleft(f"Inference error: {e}")

# =========================
# Main UI
# =========================
left, right = st.columns([2, 1])

with left:
    st.subheader("Realtime Signal")
    channels = []
    if show_gx:
        channels.append("gx")
    if show_gy:
        channels.append("gy")
    if show_gz:
        channels.append("gz")
    if show_B:
        channels.append("B")

    if not df_plot.empty and channels:
        st.plotly_chart(make_plot(df_plot, channels), use_container_width=True)
    else:
        st.write("Waiting for serial data...")

    st.subheader("Recent Stream Data")
    if not df.empty:
        recent_df = df.tail(20)[["t", "gx", "gy", "gz", "B", "raw"]].copy()
        recent_df["t"] = recent_df["t"].round(3)
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.write("No data received yet.")

with right:
    st.subheader("System Status")
    st.write(f"Serial running: **{st.session_state.running}**")
    st.write(f"Samples buffered: **{len(df)}**")
    st.write(f"Window length: **{window_sec:.1f}s**")

    if st.session_state.messages:
        for msg in st.session_state.messages:
            st.write(f"- {msg}")

    st.subheader("Current Prediction")
    if prediction_result is not None:
        pred = prediction_result["pred_3class"]

        if pred == "no_contact":
            st.success(f"Prediction: {pred}")
        elif pred == "touch":
            st.info(f"Prediction: {pred}")
        elif pred == "punch":
            st.warning(f"Prediction: {pred}")
        else:
            st.write(f"Prediction: {pred}")

        if prediction_result["prob_contact"] is not None:
            st.write("Contact Model Probabilities")
            st.json({k: round(float(v), 4) for k, v in prediction_result["prob_contact"].items()})

        if prediction_result["prob_touch_punch"] is not None:
            st.write("Touch/Punch Model Probabilities")
            st.json({k: round(float(v), 4) for k, v in prediction_result["prob_touch_punch"].items()})

        st.subheader("Current Window Features")
        feat_df = prediction_result["features"].T.reset_index()
        feat_df.columns = ["feature", "value"]
        st.dataframe(feat_df, use_container_width=True, height=300)

    else:
        st.write("Waiting for enough samples for inference...")

    st.subheader("Prediction History")
    if st.session_state.pred_history:
        hist_df = pd.DataFrame(list(st.session_state.pred_history))
        hist_df["relative_t"] = hist_df["timestamp"] - hist_df["timestamp"].iloc[0]
        hist_df["relative_t"] = hist_df["relative_t"].round(3)
        show_cols = [c for c in ["relative_t", "pred_3class"] if c in hist_df.columns]
        prob_cols = [c for c in hist_df.columns if c not in {"timestamp", "relative_t", "pred_3class"}]
        st.dataframe(hist_df[show_cols + prob_cols], use_container_width=True, height=250)
    else:
        st.write("No predictions yet.")

st.caption("This UI performs realtime three-class inference using two trained sklearn models.")

if st.session_state.running:
    time.sleep(refresh_sec)
    st.rerun()