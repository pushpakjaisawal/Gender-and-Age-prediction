"""Age & Gender Prediction Streamlit App.

A web application that predicts age group and gender from face images
using a MobileNetV2-based multi-task model trained on the UTKFace dataset.

Features:
    - Image upload with face detection and prediction
    - Live webcam prediction via streamlit-webrtc

Usage:
    Run locally: streamlit run app.py
    Deploy on Streamlit Cloud with the provided requirements.txt.

Dependencies:
    - TensorFlow >= 2.16.0 (Keras 3.x for quantization_config support)
    - OpenCV (face detection via Haar cascades)
    - streamlit-webrtc (live webcam support)
"""

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json
import threading

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Age & Gender Prediction",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("👤 Age & Gender Prediction")
st.markdown("Upload an image or use your **webcam** for live prediction.")

# ===============================
# AGE LABELS (must match training)
# ===============================
AGE_LABELS = [
    "0-3", "4-8", "9-15", "16-22",
    "23-30", "31-40", "41-50", "51-60", "61+"
]

# ===============================
# SAFE MODEL LOADER
# ===============================

def _strip_quantization_config(model_path: str) -> None:
    """Patch the .keras zip to remove quantization_config from layer configs.

    Keras 3.x saves ``'quantization_config': None`` inside Dense layer configs.
    Some deployment environments (older Keras builds on Streamlit Cloud) raise
    ``TypeError: Unrecognized keyword arguments passed to Dense`` when they
    encounter this key.  This function opens the ``model.keras`` bundle (which
    is a standard zip file), reads ``config.json``, strips every occurrence of
    ``quantization_config`` from the layer config dictionaries, writes the
    patched JSON back, and re-zips the bundle so that ``load_model`` can
    proceed without errors.

    Args:
        model_path: Path to the ``.keras`` model file to patch in-place.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        KeyError: If the expected internal structure of the ``.keras`` archive
            is missing (e.g. no ``config.json``).
    """
    import zipfile
    import tempfile
    import shutil
    import os

    abs_path = os.path.abspath(model_path)

    # Read config.json from the .keras zip
    with zipfile.ZipFile(abs_path, "r") as zf:
        if "config.json" not in zf.namelist():
            raise KeyError(
                "config.json not found inside the .keras archive. "
                "The file may be corrupted or in an unsupported format."
            )
        config_bytes = zf.read("config.json")

    config_str = config_bytes.decode("utf-8")
    config = json.loads(config_str)

    # Recursively remove 'quantization_config' from all layer configs
    def _remove_qc(obj):
        """Remove quantization_config key from a nested dict/list."""
        if isinstance(obj, dict):
            obj.pop("quantization_config", None)
            for value in obj.values():
                _remove_qc(value)
        elif isinstance(obj, list):
            for item in obj:
                _remove_qc(item)

    _remove_qc(config)

    patched_bytes = json.dumps(config, indent=2).encode("utf-8")

    # Rebuild the .keras zip with the patched config.json
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".keras")
    os.close(tmp_fd)

    try:
        with zipfile.ZipFile(abs_path, "r") as zf_in:
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
                for item in zf_in.namelist():
                    if item == "config.json":
                        zf_out.writestr(item, patched_bytes)
                    else:
                        zf_out.writestr(item, zf_in.read(item))
        # Replace original with patched version
        shutil.move(tmp_path, abs_path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


@st.cache_resource
def load_model_once():
    """Load the Keras model with quantization_config compatibility.

    First attempts a standard ``load_model`` call.  If that fails due to
    ``quantization_config`` (a Keras 3.x serialization artifact), the
    function patches the ``.keras`` archive in-place and retries.

    Returns:
        tf.keras.Model: The loaded multi-task prediction model with
            ``gender_output`` (sigmoid) and ``age_output`` (softmax) heads.
    """
    model_path = "models/best_model_finetuned.keras"

    try:
        model = load_model(model_path)
        return model
    except (TypeError, KeyError) as e:
        error_msg = str(e)
        if "quantization_config" in error_msg or "Unrecognized keyword" in error_msg:
            st.warning(
                "Patching model file for Keras compatibility…"
            )
            _strip_quantization_config(model_path)
            model = load_model(model_path)
            return model
        raise


with st.spinner("Loading model..."):
    model = load_model_once()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
st.success("Model loaded successfully! ✅")

# ===============================
# PREDICTION FUNCTION
# ===============================

def predict_face(face_img: np.ndarray):
    """Predict gender and age group for a single cropped face image.

    Preprocesses the input image to match the model's expected format
    (224×224 RGB, normalized to [0, 1]), runs inference, and decodes
    the raw output tensors into human-readable labels.

    Args:
        face_img: Cropped face region as a NumPy array in RGB order
            (shape: ``(H, W, 3)`` with dtype ``uint8``).

    Returns:
        tuple[str, float, str, float]: A 4-tuple containing:

            - **gender** (str): Either ``"Male"`` or ``"Female"``.
            - **gender_conf** (float): Confidence score for the predicted
              gender in the range [0, 1].
            - **age_group** (str): One of 9 age-range labels, e.g.
              ``"23-30"``.
            - **age_conf** (float): Softmax probability for the predicted
              age group in the range [0, 1].
    """
    img = cv2.resize(face_img, (224, 224))
    img = img / 255.0
    img_input = np.expand_dims(img, axis=0)

    gender_pred, age_pred = model.predict(img_input, verbose=0)

    gender = "Male" if gender_pred[0][0] < 0.5 else "Female"
    gender_conf = float(gender_pred[0][0]) if gender == "Male" else float(1 - gender_pred[0][0])
    age_idx = int(np.argmax(age_pred[0]))
    age_group = AGE_LABELS[age_idx]
    age_conf = float(age_pred[0][age_idx])

    return gender, gender_conf, age_group, age_conf


def process_frame(frame: np.ndarray) -> np.ndarray:
    """Detect faces in a BGR frame and annotate each with predictions.

    Converts the input frame to grayscale for Haar-cascade face detection,
    then for each detected face crops the RGB region, runs prediction, and
    draws a bounding box with a label overlay.

    Args:
        frame: Video frame in BGR format (OpenCV default),
            shape ``(H, W, 3)`` with dtype ``uint8``.

    Returns:
        np.ndarray: Annotated frame in RGB format with bounding boxes
            and prediction labels drawn on detected faces.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = rgb[y:y+h, x:x+w]

        gender, g_conf, age_group, a_conf = predict_face(face)

        label = f"{gender} ({g_conf:.0%}), {age_group} ({a_conf:.0%})"

        # Draw bounding box
        cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(rgb, (x, y-30), (x+w, y), (0, 255, 0), -1)
        cv2.putText(rgb, label, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return rgb


# ===============================
# TAB 1: IMAGE UPLOAD
# ===============================
tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Live Webcam"])

with tab1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        # Read and display
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process
        result = process_frame(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        st.image(result, use_container_width=True)

        if len(faces) > 0:
            st.markdown(f"### Detected {len(faces)} face(s)")
            for i, (x, y, w, h) in enumerate(faces):
                face = img_rgb[y:y+h, x:x+w]
                gender, g_conf, age_group, a_conf = predict_face(face)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Gender", gender, delta=f"{g_conf:.0%} confidence")
                with col2:
                    st.metric("Age Group", age_group, delta=f"{a_conf:.0%} confidence")
                with col3:
                    st.image(face, width=100)
        else:
            st.warning("No faces detected in this image.")


# ===============================
# TAB 2: LIVE WEBCAM (streamlit-webrtc)
# ===============================
with tab2:
    st.header("Live Webcam Prediction")
    st.warning(
        "Click **Start** to begin webcam detection. "
        "Click **Stop** to end the session."
    )

    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

        class FaceDetector(VideoProcessorBase):
            """Real-time face detection and age/gender prediction processor.

            Subclasses ``VideoProcessorBase`` from *streamlit-webrtc* to
            receive each video frame, run face detection and prediction,
            and return the annotated frame for display.

            Attributes:
                _lock: Threading lock to prevent concurrent prediction calls
                    on the shared Keras model.
            """

            def __init__(self):
                """Initialize the processor with a threading lock."""
                self._lock = threading.Lock()

            def recv(self, frame):
                """Process a single video frame from the webcam stream.

                Converts the frame from BGR to a NumPy array, runs face
                detection and prediction, and returns an annotated RGB frame.

                Args:
                    frame: WebRTC video frame object from streamlit-webrtc.

                Returns:
                    av.VideoFrame: Annotated frame in RGB24 format.
                """
                img = frame.to_ndarray(format="bgr24")
                result = process_frame(img)
                return av.VideoFrame.from_ndarray(result, format="rgb24")

        webrtc_ctx = webrtc_streamer(
            key="age-gender-detection",
            video_processor_factory=FaceDetector,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
        )

        if webrtc_ctx.state.playing:
            st.info("🔥 Webcam is active! Detection running...")

    except ImportError:
        st.error(
            "**streamlit-webrtc** is not installed. "
            "Webcam mode only works when deployed on Streamlit Cloud or local server.\n\n"
            "Install it with: `pip install streamlit-webrtc`"
        )
        st.info(
            "💡 **Tip:** Use **Upload Image** tab above for now, "
            "or deploy this app to Streamlit Cloud for full webcam support."
        )


# ===============================
# SIDEBAR INFO
# ===============================
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "This app predicts **age group** and **gender** "
        "from face images using a MobileNetV2-based model "
        "trained on the UTKFace dataset."
    )
    st.header("Model Details")
    st.markdown(
        "- **Backbone:** MobileNetV2 (ImageNet)\n"
        "- **Input Size:** 224 x 224\n"
        "- **Age Groups:** 9 classes\n"
        "- **Gender:** Binary (Male/Female)\n"
        "- **Accuracy:** See training notebook"
    )
    st.header("Age Groups")
    for label in AGE_LABELS:
        st.markdown(f"- {label}")
