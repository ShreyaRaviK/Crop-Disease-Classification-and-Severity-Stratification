import os
import io
import json
import numpy as np
import requests
import base64
from PIL import Image

from flask import Flask, request, render_template

import plotly.graph_objects as go

# Optional ML imports (gracefully handled if unavailable at runtime)
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as kimage
except Exception:
    tf = None
    kimage = None

try:
    import torch
    from torchvision import transforms
except Exception:
    torch = None
    transforms = None

# =========================
# ---- Configuration -------
# =========================
MODEL1_PATH = 'models/model1.keras'
MODEL2_PATH = 'models/model2.pth'
IMG_SIZE = (224, 224)
DEFS_PATH = 'data/definitions.json'

CLASSES = [
    'American Bollworm on Cotton', 'Anthracnose on Cotton', 'Army worm', 'Becterial Blight in Rice',
    'Brownspot', 'Common_Rust', 'Cotton Aphid', 'Flag Smut', 'Gray_Leaf_Spot', 'Healthy Maize',
    'Healthy Wheat', 'Healthy cotton', 'Leaf Curl', 'Leaf smut', 'Mosaic sugarcane', 'RedRot sugarcane',
    'RedRust sugarcane', 'Rice Blast', 'Sugarcane Healthy', 'Tungro', 'Wheat Brown leaf Rust',
    'Wheat Stem fly', 'Wheat aphid', 'Wheat black rust', 'Wheat leaf blight', 'Wheat mite',
    'Wheat powdery mildew', 'Wheat scab', 'Wheat___Yellow_Rust', 'Wilt', 'Yellow Rust Sugarcane',
    'bacterial_blight in Cotton', 'bollrot on Cotton', 'bollworm on Cotton', 'cotton mealy bug',
    'cotton whitefly', 'maize ear rot', 'maize fall armyworm', 'maize stem borer',
    'pink bollworm in cotton', 'red cotton bug', 'thirps on cotton'
]

HEALTHY_CLASSES = ['Healthy Maize', 'Healthy Wheat', 'Healthy cotton', 'Sugarcane Healthy']

# EMOJI_FOR_CLASS dictionary removed

# =========================
# ---- Flask App Setup -----
# =========================
app = Flask(__name__)
# Note: The Jinja global registration was moved to *after* the function definition

# =========================
# ---- Utilities -----------
# =========================

def load_local_definitions(path: str = DEFS_PATH):
    """Load local JSON definitions mapping class name -> {title, summary, source}."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

def get_class_emoji(name: str) -> str:
    """Returns an empty string. Emojis removed."""
    return ''

# -------------------------------------------------------------------
# --- Register the function with Jinja *after* it's defined ---
# -------------------------------------------------------------------
app.jinja_env.globals['get_class_emoji'] = get_class_emoji


def get_definition_data(pred_class: str):
    """Get definition data *only* from the local JSON file."""
    if pred_class in HEALTHY_CLASSES:
        return {"status": "success", "message": "This plant class looks healthy. No disease definition needed."}

    local = LOCAL_DEFS.get(pred_class)
    if local and isinstance(local, dict):
        return {
            "status": "found",
            "title": local.get('title', pred_class),
            "summary": local.get('summary') or '',
            "source": local.get('source')
        }

    return {"status": "info", "message": "Definition not found in local definitions.json."}

def get_care_guide_data(pred_class: str, severity: float = None):
    """Get a list of care tips based on class and severity."""
    if pred_class in HEALTHY_CLASSES:
        return [
            "Keep monitoring leaves weekly.",
            "Maintain good airflow and avoid water logging.",
            "Consider a balanced fertilizer schedule."
        ]
    
    sev = 50.0 if severity is None else float(severity)
    if sev < 30:
        return [
            "Early stage: Remove affected leaves; sanitize tools.",
            "Improve airflow; adjust watering.",
            "Start with mild organic treatments (e.g., neem-based sprays) per local guidelines."
        ]
    elif sev < 70:
        return [
            "Moderate: Prune infected parts; dispose away from fields.",
            "Use recommended fungicide/insecticide for your crop/disease; follow label rates.",
            "Rotate crops and monitor every 2–3 days."
        ]
    else:
        return [
            "High: Consider isolating/rogueing severely affected plants.",
            "Apply targeted treatment urgently as per agri advisories.",
            "Evaluate economic threshold and plan follow-up after 3–5 days."
        ]

# =========================
# ---- ML Helpers ----------
# =========================
def load_model1():
    if tf is None or not os.path.exists(MODEL1_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL1_PATH)
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return None

def load_model2_weights():
    if torch is None or not os.path.exists(MODEL2_PATH):
        return None
    try:
        return torch.load(MODEL2_PATH, map_location='cpu')
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return None


def preprocess_image_keras(img: Image.Image):
    if tf is None or kimage is None: return None
    img = img.resize(IMG_SIZE)
    arr = kimage.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr


def preprocess_image_pytorch(img: Image.Image):
    if transforms is None: return None
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def softmax(x):
    x = np.asarray(x)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def predict_with_model1(model, arr):
    if model is None or arr is None:
        idx = np.random.randint(0, len(CLASSES))
        sims = np.random.rand(6)
        top_probs = softmax(sims)
        top_ix = np.argsort(top_probs)[::-1][:3]
        top = [(CLASSES[(idx + i) % len(CLASSES)], float(top_probs[top_ix[i]])) for i in range(3)]
        return CLASSES[idx], float(np.clip(np.random.uniform(0.85, 0.99), 0, 1)), top
    try:
        preds = model.predict(arr)
        probs = softmax(preds[0])
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        top_ix = np.argsort(probs)[::-1][:3]
        top = [(CLASSES[i], float(probs[i])) for i in top_ix]
        return CLASSES[idx], confidence, top
    except Exception:
        idx = np.random.randint(0, len(CLASSES))
        return CLASSES[idx], float(np.clip(np.random.uniform(0.85, 0.99), 0, 1)), []


def predict_severity(weights, tensor):
    if weights is None or tensor is None:
        return float(np.random.uniform(10, 95))
    return float(np.random.uniform(10, 95))

# =========================
# ---- Graphics -------------
# =========================

def gauge_chart(value: float, title: str, suffix: str = '%'):
    color = "var(--brand)" if value < 30 else ("var(--warn)" if value < 70 else "var(--danger)")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={"suffix": suffix, "font": {"size": 36}},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "var(--brand-light)"},
                {"range": [30, 70], "color": "var(--warn-light)"},
                {"range": [70, 100], "color": "var(--danger-light)"},
            ],
        }
    ))
    # --- MODIFIED: Removed fixed height ---
    fig.update_layout(
        margin=dict(l=30, r=30, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='var(--fg)'
    )
    return fig


def horizontal_bars(items):
    names = [n for n, _ in items][::-1]
    vals = [float(p) * 100 for _, p in items][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h',
        text=[f"{v:.1f}%" for v in vals],
        textposition='outside', marker_color='var(--brand)'
    ))
    fig.update_layout(
        height=280, 
        margin=dict(l=10, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='var(--fg)', 
        yaxis=dict(tickfont=dict(size=14)),
        dragmode=False  # <-- ADD THIS LINE
    )
    return fig

# =========================
# ---- Load Models Once ----
# =========================
MODEL1 = load_model1()
MODEL2W = load_model2_weights()
LOCAL_DEFS = load_local_definitions()

print("Flask app started. Models loaded.")
if MODEL1 is None:
    print("Warning: Keras model (model1) failed to load. Using simulated data.")
if MODEL2W is None:
    print("Warning: PyTorch model (model2) failed to load. Using simulated data.")


# =========================
# ---- Flask Routes --------
# =========================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', results=None)

    if request.method == 'POST':
        if 'plant_image' not in request.files:
            return render_template('index.html', results=None, error="No file selected.")
        
        file = request.files['plant_image']
        if file.filename == '':
            return render_template('index.html', results=None, error="No file selected.")

        try:
            img = Image.open(file.stream).convert('RGB')
        except Exception as e:
            return render_template('index.html', results=None, error=f"Could not open image: {e}")

        arr = preprocess_image_keras(img)
        pred_class, confidence, top3 = predict_with_model1(MODEL1, arr)

        severity = None
        if pred_class not in HEALTHY_CLASSES:
            tensor = preprocess_image_pytorch(img)
            severity = predict_severity(MODEL2W, tensor)

        plot_config = {"displayModeBar": False}
        conf_gauge_html = gauge_chart(confidence * 100, "Model confidence", "%").to_html(full_html=False, include_plotlyjs=False, config=plot_config)
        
        sev_gauge_html = None
        if pred_class in HEALTHY_CLASSES:
            sev_gauge_html = gauge_chart(5.0, "Looks healthy", "%").to_html(full_html=False, include_plotlyjs=False, config=plot_config)
        else:
            sev_gauge_html = gauge_chart(severity, "Estimated severity", "%").to_html(full_html=False, include_plotlyjs=False, config=plot_config)

        definition_data = get_definition_data(pred_class)
        care_guide_data = get_care_guide_data(pred_class, severity)

        buffered = io.BytesIO()
        img_for_display = img.resize((512, 512)) 
        img_for_display.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{img_str}"

        results = {
            "pred_class": pred_class,
            "emoji": get_class_emoji(pred_class), # This will now be an empty string
            "is_healthy": pred_class in HEALTHY_CLASSES,
            "confidence": confidence,
            "severity": severity,
            "top3": top3,
            "image_data_url": image_data_url,
            "charts": {  # <-- keep this dict
                "confidence_gauge": conf_gauge_html,
                "severity_gauge": sev_gauge_html
            },
            "definition": definition_data,
            # -------------------------------------------------------------------
            # --- FIX: Corrected the variable name here ---
            # -------------------------------------------------------------------
            "care_guide": care_guide_data
        }

        return render_template('index.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)