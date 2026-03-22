#!/usr/bin/env bash

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_BIN=""
APP_FILE="$PROJECT_DIR/realtime_inference_app.py"
REQ_FILE="$PROJECT_DIR/requirements.txt"

MODEL1="$PROJECT_DIR/outputs/contact_vs_no_contact_logreg.joblib"
MODEL2="$PROJECT_DIR/outputs/touch_vs_punch_logreg.joblib"

echo "Project directory: $PROJECT_DIR"

# --------------------------
# 1. Find Python
# --------------------------
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Error: Python not found. Please install Python 3 first."
    exit 1
fi

echo "Using Python: $PYTHON_BIN"

# --------------------------
# 2. Create venv if needed
# --------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# --------------------------
# 3. Activate venv
# --------------------------
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

# --------------------------
# 4. Install dependencies
# --------------------------
if [ -f "$REQ_FILE" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r "$REQ_FILE"
else
    echo "requirements.txt not found, installing default dependencies..."
    pip install streamlit pyserial pandas plotly joblib numpy scikit-learn shap matplotlib
fi

# --------------------------
# 5. Check app file
# --------------------------
if [ ! -f "$APP_FILE" ]; then
    echo "Error: realtime_inference_app.py not found at:"
    echo "  $APP_FILE"
    exit 1
fi

# --------------------------
# 6. Check model files
# --------------------------
if [ ! -f "$MODEL1" ]; then
    echo "Warning: model file not found:"
    echo "  $MODEL1"
fi

if [ ! -f "$MODEL2" ]; then
    echo "Warning: model file not found:"
    echo "  $MODEL2"
fi

# --------------------------
# 7. Export default model paths for Streamlit app
# --------------------------
export MODEL_CONTACT_PATH="$MODEL1"
export MODEL_TP_PATH="$MODEL2"

# --------------------------
# 8. Launch app
# --------------------------
echo "Starting Streamlit app..."
echo "If the browser does not open automatically, use the local URL shown below."
streamlit run "$APP_FILE"