import math
import queue
import threading
import time
from collections import deque
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import serial
import serial.tools.list_ports
import streamlit as st


# =========================
# Page Setup
# =========================
st.set_page_config(page_title="Magnetic Tactile Monitor", layout="wide")


# =========================
# Utility Functions
# =========================
def list_serial_ports():
    return [p.device for p in serial.tools.list_ports.comports()]


def parse_line(line: str):
    """
    Supported input formats:
      gx,gy,gz,B
      gx,gy,gz
    Returns:
      dict or None
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
        time.sleep(2.0)  # allow Arduino reset after serial open

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


def ensure_data_dir():
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def export_labeled_data(records, filename_prefix="magnetic_tactile"):
    if not records:
        return None

    df = pd.DataFrame(records)
    if df.empty:
        return None

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_path = ensure_data_dir() / f"{filename_prefix}_{ts}.csv"
    df.to_csv(output_path, index=False)
    return output_path


def make_plot(df_plot: pd.DataFrame, channels: list[str]):
    fig = go.Figure()
    for ch in channels:
        fig.add_trace(
            go.Scatter(
                x=df_plot["t"],
                y=df_plot[ch],
                mode="lines",
                name=ch,
            )
        )

    fig.update_layout(
        height=500,
        xaxis_title="Time (s)",
        yaxis_title="Magnetic reading (gauss)",
        margin=dict(l=20, r=20, t=30, b=20),
        legend_title="Channel",
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
    st.session_state.messages = deque(maxlen=30)

if "capture_enabled" not in st.session_state:
    st.session_state.capture_enabled = False

if "labeled_records" not in st.session_state:
    st.session_state.labeled_records = []

if "current_label" not in st.session_state:
    st.session_state.current_label = "no_contact"

if "last_saved_path" not in st.session_state:
    st.session_state.last_saved_path = None


# =========================
# Sidebar Controls
# =========================
st.title("Magnetic Tactile Sensor Web UI")

with st.sidebar:
    st.header("Serial Connection")

    ports = list_serial_ports()
    if ports:
        port = st.selectbox("Serial Port", ports)
    else:
        port = st.selectbox("Serial Port", [""])

    baudrate = st.selectbox("Baud Rate", [9600, 19200, 38400, 57600, 115200], index=4)
    refresh_sec = st.slider("Refresh Interval (sec)", 0.1, 2.0, 0.3, 0.1)
    visible_points = st.slider("Visible Points", 100, 2000, 400, 50)

    st.header("Channels")
    show_gx = st.checkbox("Show gx", True)
    show_gy = st.checkbox("Show gy", True)
    show_gz = st.checkbox("Show gz", True)
    show_B = st.checkbox("Show B", True)

    st.header("Labeling")
    selected_label = st.selectbox("Current Label", ["no_contact", "touch", "punch"], index=0)
    st.session_state.current_label = selected_label

    cap_col1, cap_col2 = st.columns(2)
    start_capture = cap_col1.button("Start Capture", use_container_width=True)
    stop_capture = cap_col2.button("Stop Capture", use_container_width=True)

    if start_capture:
        st.session_state.capture_enabled = True
        st.session_state.messages.appendleft(f"Capture started with label: {st.session_state.current_label}")

    if stop_capture:
        st.session_state.capture_enabled = False
        st.session_state.messages.appendleft("Capture stopped.")

    st.header("Connection Control")
    conn_col1, conn_col2 = st.columns(2)
    start_serial = conn_col1.button("Connect", use_container_width=True)
    stop_serial = conn_col2.button("Disconnect", use_container_width=True)

    if start_serial and not st.session_state.running:
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

    if stop_serial and st.session_state.running:
        st.session_state.stop_event.set()
        st.session_state.running = False
        st.session_state.messages.appendleft("Serial disconnected.")

    st.header("Data Export")
    filename_prefix = st.text_input("Filename Prefix", value="magnetic_tactile")
    save_csv = st.button("Save CSV", use_container_width=True)
    clear_data = st.button("Clear Captured Data", use_container_width=True)

    if save_csv:
        path = export_labeled_data(st.session_state.labeled_records, filename_prefix=filename_prefix)
        if path:
            st.session_state.last_saved_path = str(path)
            st.session_state.messages.appendleft(f"Saved CSV: {path}")
        else:
            st.session_state.messages.appendleft("No labeled data to save.")

    if clear_data:
        st.session_state.labeled_records = []
        st.session_state.messages.appendleft("Captured labeled data cleared.")


# =========================
# Main Layout
# =========================
status_col, metric_col = st.columns([2, 1])

with status_col:
    st.subheader("Status")
    st.write(f"Serial running: **{st.session_state.running}**")
    st.write(f"Capture enabled: **{st.session_state.capture_enabled}**")
    st.write(f"Current label: **{st.session_state.current_label}**")

    if st.session_state.last_saved_path:
        st.success(f"Last saved file: {st.session_state.last_saved_path}")

    if st.session_state.messages:
        for msg in st.session_state.messages:
            st.write(f"- {msg}")

# Drain serial queue
while not st.session_state.data_queue.empty():
    item = st.session_state.data_queue.get()

    if "error" in item:
        st.session_state.messages.appendleft(item["error"])
        st.session_state.running = False
        continue

    st.session_state.buffer.append(item)

    if st.session_state.capture_enabled:
        labeled_item = dict(item)
        labeled_item["label"] = st.session_state.current_label
        st.session_state.labeled_records.append(labeled_item)

# Build DataFrame
df = pd.DataFrame(list(st.session_state.buffer))

if not df.empty:
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    df_plot = df.tail(visible_points).copy()
else:
    df_plot = pd.DataFrame(columns=["t", "gx", "gy", "gz", "B"])

with metric_col:
    st.subheader("Live Values")
    if not df.empty:
        latest = df.iloc[-1]
        st.metric("gx", f"{latest['gx']:.4f}")
        st.metric("gy", f"{latest['gy']:.4f}")
        st.metric("gz", f"{latest['gz']:.4f}")
        st.metric("B", f"{latest['B']:.4f}")
        st.metric("Buffered Samples", len(df))
        st.metric("Captured Samples", len(st.session_state.labeled_records))
    else:
        st.info("No data yet.")

# Plot
st.subheader("Realtime Plot")

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
    fig = make_plot(df_plot, channels)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Waiting for serial data...")

# Recent streamed data
st.subheader("Recent Stream Data")
if not df.empty:
    recent_df = df.tail(20)[["t", "gx", "gy", "gz", "B", "raw"]].copy()
    recent_df["t"] = recent_df["t"].round(3)
    st.dataframe(recent_df, use_container_width=True)
else:
    st.write("No serial data received yet.")

# Recent labeled data
st.subheader("Captured Labeled Data")
if st.session_state.labeled_records:
    labeled_df = pd.DataFrame(st.session_state.labeled_records).tail(20).copy()
    if "timestamp" in labeled_df.columns:
        labeled_df["relative_t"] = labeled_df["timestamp"] - labeled_df["timestamp"].iloc[0]
        labeled_df["relative_t"] = labeled_df["relative_t"].round(3)

    show_cols = [c for c in ["relative_t", "gx", "gy", "gz", "B", "label"] if c in labeled_df.columns]
    st.dataframe(labeled_df[show_cols], use_container_width=True)
else:
    st.write("No labeled samples captured yet.")

st.caption("Make sure the Arduino baud rate matches the selected baud rate in the sidebar.")

if st.session_state.running:
    time.sleep(refresh_sec)
    st.rerun()