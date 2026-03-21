import streamlit as st
import numpy as np
import cv2
import json
import joblib
import os
import gdown
from pathlib import Path
from skimage.feature import hog

# ======================================================
# Page config
# ======================================================
st.set_page_config(
    page_title="Sports Ball Classifier",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# Global CSS — Dark sports-editorial theme
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0a0f;
    color: #e8e4dc;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1200px; }

[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #e8e4dc !important; }

.sidebar-logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #f5c542 !important;
    line-height: 1.1;
    padding: 0.5rem 0 1rem;
}
.sidebar-logo span { color: #e8e4dc !important; font-weight: 300; }

.stat-card {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.stat-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b6680;
    margin-bottom: 4px;
}
.stat-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f5c542;
}

.page-header { margin-bottom: 2rem; padding-bottom: 1.2rem; border-bottom: 1px solid #1e1e2e; }
.page-eyebrow {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #f5c542;
    font-weight: 600;
    margin-bottom: 6px;
}
.page-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    text-transform: uppercase;
    line-height: 1;
    color: #e8e4dc;
    letter-spacing: 0.02em;
}

.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #e8e4dc;
    padding: 0.8rem 0 0.5rem;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 1rem;
    margin-top: 1.2rem;
}

.info-card {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.4rem;
    height: 100%;
    margin-bottom: 1rem;
}
.info-card-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #f5c542;
    margin-bottom: 0.8rem;
}
.info-card p, .info-card li {
    font-size: 0.88rem;
    color: #9993a8;
    line-height: 1.7;
}

.result-main {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a45;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.2rem;
}
.result-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #6b6680;
    margin-bottom: 8px;
}
.result-class {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    text-transform: uppercase;
    color: #f5c542;
    letter-spacing: 0.05em;
    line-height: 1.1;
}
.result-conf { font-size: 1rem; color: #9993a8; margin-top: 4px; }

.pred-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.pred-rank { font-family: 'Barlow Condensed', sans-serif; font-size: 0.75rem; font-weight: 700; color: #6b6680; width: 20px; flex-shrink: 0; }
.pred-name { font-size: 0.85rem; color: #e8e4dc; width: 160px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pred-bar-wrap { flex: 1; height: 6px; background: #1e1e2e; border-radius: 3px; overflow: hidden; }
.pred-bar-fill { height: 100%; border-radius: 3px; background: #f5c542; }
.pred-pct { font-family: 'Barlow Condensed', sans-serif; font-size: 0.85rem; font-weight: 600; color: #9993a8; width: 42px; text-align: right; }

.arch-code {
    background: #0f0f18;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #9993a8;
    line-height: 1.8;
    white-space: pre;
    margin-bottom: 1rem;
}
.arch-code .hl { color: #f5c542; font-weight: bold; }

.ref-item {
    font-size: 0.82rem;
    color: #6b6680;
    padding: 6px 0;
    border-bottom: 1px solid #1a1a28;
    line-height: 1.5;
}
.ref-item a { color: #f5c542; text-decoration: none; }

[data-testid="stFileUploader"] {
    background: #13131f !important;
    border: 1px dashed #2a2a45 !important;
    border-radius: 12px !important;
}
[data-testid="stRadio"] label { font-size: 0.88rem !important; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# ⚠️ ใส่ Google Drive File ID ของไฟล์โมเดลที่นี่
# ======================================================
FILE_IDS = {
    "nn_model.keras":          "ใส่ FILE_ID ของ nn_model.keras",
    "ensemble_model.pkl":      "ใส่ FILE_ID ของ ensemble_model.pkl",
    "scaler.pkl":              "ใส่ FILE_ID ของ scaler.pkl",
    "dataset2_label_map.json": "ใส่ FILE_ID ของ dataset2_label_map.json",
}

MODELS_PATH  = Path("models")
MODELS_PATH.mkdir(exist_ok=True)
DATASET_NAME = 'dataset2'

def download_models():
    for filename, file_id in FILE_IDS.items():
        dest = MODELS_PATH / filename
        if not dest.exists():
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(dest), quiet=False)

with st.spinner("กำลังโหลดโมเดลจาก Google Drive..."):
    download_models()

# ======================================================
# Load models (cached)
# ======================================================
@st.cache_resource
def load_ml():
    ensemble = joblib.load(MODELS_PATH / 'ensemble_model.pkl')
    scaler   = joblib.load(MODELS_PATH / 'scaler.pkl')
    with open(MODELS_PATH / f'{DATASET_NAME}_label_map.json') as f:
        lmap = json.load(f)
    return ensemble, scaler, {v: k for k, v in lmap.items()}

@st.cache_resource
def load_nn():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODELS_PATH / 'nn_model.keras')
    with open(MODELS_PATH / f'{DATASET_NAME}_label_map.json') as f:
        lmap = json.load(f)
    return model, {v: k for k, v in lmap.items()}

# ======================================================
# Predict helpers
# ======================================================
def predict_ml(img_bytes, ensemble, scaler, i2c):
    arr  = np.frombuffer(img_bytes, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img  = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feat = hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2))
    prob = ensemble.predict_proba(scaler.transform([feat]))[0]
    idx  = int(np.argmax(prob))
    return i2c[idx], float(prob[idx]), prob

def predict_nn(img_bytes, model, i2c):
    arr  = np.frombuffer(img_bytes, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img  = cv2.resize(img, (224, 224))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    prob = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    idx  = int(np.argmax(prob))
    return i2c[idx], float(prob[idx]), prob

def show_top5(proba, i2c):
    top5 = np.argsort(proba)[::-1][:5]
    html = ""
    for rank, idx in enumerate(top5):
        name = i2c.get(idx, str(idx)).replace('_', ' ').title()
        pct  = proba[idx] * 100
        fill = "#f5c542" if rank == 0 else "#2a2a45"
        html += f"""
        <div class="pred-row">
          <div class="pred-rank">#{rank+1}</div>
          <div class="pred-name">{name}</div>
          <div class="pred-bar-wrap"><div class="pred-bar-fill" style="width:{pct:.1f}%;background:{fill};"></div></div>
          <div class="pred-pct">{pct:.1f}%</div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)

# ======================================================
# Sidebar
# ======================================================
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚽ Sports<br><span>Ball Classifier</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("เลือกหน้า", [
        "📖  อธิบาย ML Model",
        "📖  อธิบาย Neural Network",
        "🧪  ทดสอบ ML Model",
        "🧪  ทดสอบ Neural Network"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div class="stat-card">
        <div class="stat-label">Dataset</div>
        <div class="stat-value" style="font-size:1rem;color:#e8e4dc;">Sports Balls · 15 Classes</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">ML Ensemble</div>
        <div class="stat-value">34.13%</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Neural Network</div>
        <div class="stat-value" style="color:#5cb85c;">80.23%</div>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# PAGE 1 — อธิบาย ML
# ======================================================
if "ML Model" in page and "อธิบาย" in page:
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Machine Learning · Ensemble</div>
        <div class="page-title">ML Ensemble Model</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">1 · Dataset & การเตรียมข้อมูล</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    <p><b style="color:#e8e4dc;">Sports Balls Multiclass</b> จาก Kaggle — รูปลูกบอล <b style="color:#f5c542;">15 ประเภท</b> รวมประมาณ 9,000 รูป</p>
    <ul>
        <li>ตรวจสอบและลบรูปเสียหาย (Corrupt images)</li>
        <li>Resize ทุกรูปเป็น <b style="color:#f5c542;">224×224 px</b></li>
        <li>บันทึก Label Map (class → index) เป็น JSON</li>
        <li>แบ่ง Train / Validation / Test</li>
    </ul>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">2 · HOG Feature Extraction</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    <p><b style="color:#e8e4dc;">HOG (Histogram of Oriented Gradients)</b> แปลงรูปภาพเป็น vector ตัวเลขโดยจับ pattern ของขอบและทิศทางในรูป</p>
    <p>img_size=128×128 · orientations=9 · pixels_per_cell=(8,8) · cells_per_block=(2,2)<br>
    → Feature vector ขนาด <b style="color:#f5c542;">~8,100 ค่า</b> ต่อรูป</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">3 · โมเดลใน Ensemble</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, title, body in zip(
        [c1, c2, c3],
        ["🌲 Random Forest", "📐 SVM (RBF)", "⚡ XGBoost"],
        [
            "n_estimators=200 · สร้าง Decision Tree 200 ต้น แล้ว vote เสียงข้างมาก · ทนต่อ overfitting",
            "kernel='rbf' · C=10 · หา hyperplane แบ่ง class ด้วย kernel trick · probability=True",
            "n_estimators=200 · max_depth=6 · Gradient Boosting แต่ละต้นแก้ error ของต้นก่อนหน้า"
        ]
    ):
        col.markdown(f'<div class="info-card"><div class="info-card-title">{title}</div><p>{body}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">4 · VotingClassifier (Soft Voting)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    <p>รวมทั้ง 3 โมเดลด้วย <b style="color:#f5c542;">Soft Voting</b> — นำค่า probability มาเฉลี่ย แล้วเลือก class สูงสุด</p>
    <p>RF: ~28% &nbsp;·&nbsp; SVM: ~34% &nbsp;·&nbsp; XGB: ~31% &nbsp;→&nbsp; <b style="color:#f5c542;">Ensemble: 34.13%</b></p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">5 · แหล่งอ้างอิง</div>', unsafe_allow_html=True)
    for ref in [
        "Dalal & Triggs (2005). Histograms of oriented gradients for human detection. <i>CVPR 2005</i>",
        "Breiman (2001). Random Forests. <i>Machine Learning, 45(1)</i>",
        "Chen & Guestrin (2016). XGBoost. <i>KDD 2016</i>",
        "scikit-learn — <a href='https://scikit-learn.org' target='_blank'>scikit-learn.org</a>",
        "Kaggle — Sports Balls Multiclass Image Classification",
    ]:
        st.markdown(f'<div class="ref-item">{ref}</div>', unsafe_allow_html=True)

# ======================================================
# PAGE 2 — อธิบาย NN
# ======================================================
elif "Neural Network" in page and "อธิบาย" in page:
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Deep Learning · Transfer Learning</div>
        <div class="page-title">EfficientNetB0</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">1 · Dataset & Data Augmentation</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    <p>ใช้ Dataset เดียวกับ ML — Sports Balls 15 classes</p>
    <p>Data Augmentation บน train set: Rotation ±20° · Width/Height shift 10% · Horizontal flip · Zoom 10% · Rescale 0–1</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">2 · ทฤษฎี EfficientNetB0</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card">
    <p><b style="color:#e8e4dc;">EfficientNet</b> ออกแบบโดย Google (Tan & Le, 2019) ใช้ <b style="color:#f5c542;">Compound Scaling</b> — ปรับ depth, width, resolution พร้อมกันอย่างสมดุล ทำให้ได้ accuracy สูงโดยใช้ parameters น้อยกว่า ResNet/VGG มาก</p>
    <p><b style="color:#f5c542;">B0</b> คือขนาดเล็กที่สุด เหมาะกับ dataset ขนาดกลางและ fine-tuning บน task ใหม่</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">3 · โครงสร้างโมเดล</div>', unsafe_allow_html=True)
    st.markdown("""<div class="arch-code"><span class="hl">EfficientNetB0</span>  (pretrained ImageNet · frozen in Phase 1)
    ↓
<span class="hl">GlobalAveragePooling2D</span>
    ↓
<span class="hl">BatchNormalization</span>
    ↓
<span class="hl">Dense</span>(256, activation='relu')
    ↓
<span class="hl">Dropout</span>(0.5)
    ↓
<span class="hl">Dense</span>(15, activation='softmax')   ← output 15 classes</div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">4 · กลยุทธ์ Train (2 Phase)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown("""<div class="info-card">
    <div class="info-card-title">Phase 1 — Top Layer Training</div>
    <p>Freeze base model ทั้งหมด · Train เฉพาะ layer ใหม่ · lr=1e-3 · 10 epochs · ให้ layer ใหม่ปรับตัวก่อน</p>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="info-card">
    <div class="info-card-title">Phase 2 — Fine-tuning</div>
    <p>Unfreeze ทุก layer · Train ทั้งโมเดล · lr=1e-5 · 20 epochs + EarlyStopping · ได้ accuracy สูงสุด</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">5 · ผลลัพธ์</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-card" style="text-align:center;">
    <div class="stat-value" style="font-size:3rem;color:#5cb85c;">80.23%</div>
    <div class="stat-label">Val Accuracy — สูงกว่า ML Ensemble กว่า 2 เท่า</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">6 · แหล่งอ้างอิง</div>', unsafe_allow_html=True)
    for ref in [
        "Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. <i>ICML 2019</i>",
        "TensorFlow — <a href='https://www.tensorflow.org' target='_blank'>tensorflow.org</a>",
        "Keras Applications — <a href='https://keras.io/api/applications/efficientnet' target='_blank'>EfficientNet</a>",
        "Kaggle — Sports Balls Multiclass Image Classification",
    ]:
        st.markdown(f'<div class="ref-item">{ref}</div>', unsafe_allow_html=True)

# ======================================================
# PAGE 3 — ทดสอบ ML
# ======================================================
elif "ML Model" in page and "ทดสอบ" in page:
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Demo · Machine Learning Ensemble</div>
        <div class="page-title">ทดสอบ ML Model</div>
    </div>""", unsafe_allow_html=True)

    try:
        ensemble, scaler, i2c = load_ml()
        st.markdown('<p style="color:#5cb85c;font-size:0.85rem;">✔ โหลดโมเดลสำเร็จ</p>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
        st.stop()

    st.markdown("""<div class="info-card">
    <p>อัปโหลด <b style="color:#f5c542;">รูปลูกบอลกีฬา</b> เช่น ลูกฟุตบอล เทนนิส บาสเกตบอล ฯลฯ<br>
    โมเดลจะจำแนกประเภทจาก HOG features ด้วย Ensemble ของ RF + SVM + XGBoost</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("เลือกรูปภาพ (JPG / PNG)", type=['jpg','jpeg','png'], key='ml_up')

    if uploaded:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.image(uploaded, caption="รูปที่อัปโหลด", use_container_width=True)
        with col2:
            with st.spinner("กำลังวิเคราะห์..."):
                pred, conf, proba = predict_ml(uploaded.read(), ensemble, scaler, i2c)
            st.markdown(f"""
            <div class="result-main">
                <div class="result-label">ผลการทำนาย</div>
                <div class="result-class">{pred.replace('_',' ')}</div>
                <div class="result-conf">Confidence {conf*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top 5 Predictions</div>', unsafe_allow_html=True)
            show_top5(proba, i2c)

# ======================================================
# PAGE 4 — ทดสอบ NN
# ======================================================
elif "Neural Network" in page and "ทดสอบ" in page:
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Demo · EfficientNetB0</div>
        <div class="page-title">ทดสอบ Neural Network</div>
    </div>""", unsafe_allow_html=True)

    try:
        nn_model, i2c_nn = load_nn()
        st.markdown('<p style="color:#5cb85c;font-size:0.85rem;">✔ โหลดโมเดลสำเร็จ</p>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
        st.stop()

    st.markdown("""<div class="info-card">
    <p>อัปโหลด <b style="color:#f5c542;">รูปลูกบอลกีฬา</b> แล้ว EfficientNetB0 จะจำแนกประเภท<br>
    ด้วย accuracy <b style="color:#5cb85c;">80.23%</b> จาก Transfer Learning บน ImageNet</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("เลือกรูปภาพ (JPG / PNG)", type=['jpg','jpeg','png'], key='nn_up')

    if uploaded:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.image(uploaded, caption="รูปที่อัปโหลด", use_container_width=True)
        with col2:
            with st.spinner("กำลังวิเคราะห์..."):
                pred, conf, proba = predict_nn(uploaded.read(), nn_model, i2c_nn)
            st.markdown(f"""
            <div class="result-main">
                <div class="result-label">ผลการทำนาย</div>
                <div class="result-class">{pred.replace('_',' ')}</div>
                <div class="result-conf">Confidence {conf*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top 5 Predictions</div>', unsafe_allow_html=True)
            show_top5(proba, i2c_nn)
