import streamlit as st
import numpy as np
import cv2
import json
import joblib
import tempfile
import os
from pathlib import Path
from skimage.feature import hog

# ======================================================
# ตั้งค่าหน้าเว็บ
# ======================================================
st.set_page_config(
    page_title="Sports Image Classification",
    page_icon="🏆",
    layout="wide"
)

# ======================================================
# โหลดโมเดล (cache เพื่อไม่ต้องโหลดซ้ำทุกครั้ง)
# ======================================================
MODELS_PATH = Path(__file__).parent.parent / 'models'
DATASET_NAME = 'dataset2'

@st.cache_resource
def load_ml_models():
    ensemble = joblib.load(MODELS_PATH / 'ensemble_model.pkl')
    scaler   = joblib.load(MODELS_PATH / 'scaler.pkl')
    with open(MODELS_PATH / f'{DATASET_NAME}_label_map.json') as f:
        label_map = json.load(f)
    idx_to_class = {v: k for k, v in label_map.items()}
    return ensemble, scaler, idx_to_class

@st.cache_resource
def load_nn_model():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODELS_PATH / 'nn_model.keras')
    with open(MODELS_PATH / f'{DATASET_NAME}_label_map.json') as f:
        label_map = json.load(f)
    idx_to_class = {v: k for k, v in label_map.items()}
    return model, idx_to_class

# ======================================================
# ฟังก์ชัน Predict
# ======================================================
def predict_ml(img_bytes, ensemble, scaler, idx_to_class):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img   = cv2.resize(img, (128, 128))
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feat  = hog(gray, orientations=9, pixels_per_cell=(8,8),
                cells_per_block=(2,2), visualize=False)
    feat_scaled = scaler.transform([feat])
    proba       = ensemble.predict_proba(feat_scaled)[0]
    pred_idx    = int(np.argmax(proba))
    return idx_to_class[pred_idx], float(proba[pred_idx]), proba

def predict_nn(img_bytes, model, idx_to_class):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img   = cv2.resize(img, (224, 224))
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img   = img / 255.0
    img   = np.expand_dims(img, axis=0)
    proba    = model.predict(img, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    return idx_to_class[pred_idx], float(proba[pred_idx]), proba

# ======================================================
# Sidebar Navigation
# ======================================================
st.sidebar.title("🏆 Sports Classification")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "เลือกหน้า",
    [
        "📖 อธิบาย ML Model",
        "📖 อธิบาย Neural Network",
        "🧪 ทดสอบ ML Model",
        "🧪 ทดสอบ Neural Network"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("**Dataset:** Sports Balls (15 classes)\n\n**ML Accuracy:** 34.13%\n\n**NN Accuracy:** 80.23%")

# ======================================================
# หน้า 1: อธิบาย ML Model
# ======================================================
if page == "📖 อธิบาย ML Model":
    st.title("📖 ML Ensemble Model")
    st.markdown("---")

    st.header("1. Dataset และการเตรียมข้อมูล")
    st.markdown("""
    **Dataset ที่ใช้:** Sports Balls Multiclass Image Classification (Kaggle)
    - จำนวน **15 class** ได้แก่ อเมริกันฟุตบอล, เบสบอล, บาสเกตบอล, คริกเก็ต, ฟุตบอล, กอล์ฟ, ฮอกกี้, ลาครอส, รักบี้, สนุกเกอร์, เทนนิส, วอลเลย์บอล ฯลฯ
    - รูปทั้งหมด: **~9,000 รูป**
    
    **ขั้นตอนเตรียมข้อมูล:**
    1. ตรวจสอบและลบรูปที่เสียหาย (Corrupt images)
    2. Resize ทุกรูปเป็น **224×224 px**
    3. บันทึก Label Map (class → index) เป็น JSON
    """)

    st.header("2. การดึง Features ด้วย HOG")
    st.markdown("""
    **HOG (Histogram of Oriented Gradients)** คือเทคนิคดึง feature จากรูปภาพ  
    โดยจับ pattern ของขอบ (edge) และทิศทางในรูป แบ่งรูปเป็น cell เล็กๆ แล้วนับทิศทางของ gradient

    **ค่าที่ใช้:**
    - `img_size` = 128×128
    - `orientations` = 9
    - `pixels_per_cell` = (8, 8)
    - `cells_per_block` = (2, 2)
    - Feature vector ขนาด: **~8,100 ค่า** ต่อรูป
    """)

    st.header("3. ทฤษฎีของแต่ละโมเดล")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🌲 Random Forest")
        st.markdown("""
        - สร้าง Decision Tree หลายต้น (200 ต้น)
        - แต่ละต้นเห็น subset ของ features
        - รวมผลด้วยการ vote เสียงข้างมาก
        - ทนต่อ overfitting ได้ดี
        """)
    with col2:
        st.subheader("📐 SVM")
        st.markdown("""
        - หา hyperplane ที่แบ่ง class ได้ดีที่สุด
        - ใช้ RBF kernel สำหรับข้อมูลที่ไม่ linear
        - C=10, gamma='scale'
        - เปิด probability=True เพื่อใช้ใน ensemble
        """)
    with col3:
        st.subheader("⚡ XGBoost")
        st.markdown("""
        - สร้าง Tree แบบ Gradient Boosting
        - แต่ละต้นแก้ error ของต้นก่อนหน้า
        - 200 estimators, max_depth=6
        - learning_rate=0.1
        """)

    st.header("4. Ensemble ด้วย VotingClassifier")
    st.markdown("""
    รวมทั้ง 3 โมเดลด้วย **Soft Voting** คือนำค่า probability ของแต่ละโมเดลมาเฉลี่ย  
    แล้วเลือก class ที่ได้ค่าเฉลี่ยสูงสุด ทำให้แม่นยำกว่าการใช้โมเดลเดียว

    | โมเดล | Accuracy |
    |---|---|
    | Random Forest | ~28% |
    | SVM | ~34% |
    | XGBoost | ~31% |
    | **Ensemble** | **34.13%** |
    """)

    st.header("5. แหล่งอ้างอิง")
    st.markdown("""
    - Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR 2005*
    - Breiman, L. (2001). Random Forests. *Machine Learning, 45(1), 5–32*
    - Chen, T., & Guestrin, C. (2016). XGBoost. *KDD 2016*
    - scikit-learn documentation: https://scikit-learn.org
    - Kaggle Dataset: Sports Balls Multiclass Image Classification
    """)

# ======================================================
# หน้า 2: อธิบาย Neural Network
# ======================================================
elif page == "📖 อธิบาย Neural Network":
    st.title("📖 Neural Network — EfficientNetB0")
    st.markdown("---")

    st.header("1. Dataset และการเตรียมข้อมูล")
    st.markdown("""
    ใช้ Dataset เดียวกับ ML Model (Sports Balls, 15 classes)  
    
    **Data Augmentation** เพิ่มความหลากหลายของข้อมูล:
    - Rotation ±20°
    - Width/Height shift 10%
    - Horizontal flip
    - Zoom 10%
    - Rescale pixel 0-255 → 0-1
    """)

    st.header("2. ทฤษฎี EfficientNetB0")
    st.markdown("""
    **EfficientNet** คือสถาปัตยกรรม CNN ที่ออกแบบโดย Google (2019)  
    ใช้วิธี **Compound Scaling** ปรับ depth, width, resolution พร้อมกันอย่างสมดุล  
    ทำให้ได้ accuracy สูงโดยใช้ parameters น้อยกว่า ResNet และ VGG มาก

    **B0** คือขนาดเล็กที่สุดในตระกูล EfficientNet เหมาะสำหรับ:
    - Dataset ขนาดกลาง
    - ทรัพยากรจำกัด (ไม่มี GPU แรงๆ)
    - Fine-tuning บน task ใหม่
    """)

    st.header("3. โครงสร้างโมเดล")
    st.code("""
EfficientNetB0 (pretrained ImageNet, frozen)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(15, activation='softmax')   ← 15 classes
    """)

    st.header("4. กลยุทธ์การ Train (Transfer Learning)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Phase 1 — Top Layer Training")
        st.markdown("""
        - Freeze ทุก layer ของ EfficientNetB0
        - Train เฉพาะ layer ที่เราต่อเพิ่ม
        - Learning rate = 1e-3
        - 10 epochs
        - ทำให้ layer ใหม่ปรับตัวก่อน
        """)
    with col2:
        st.subheader("Phase 2 — Fine-tuning")
        st.markdown("""
        - Unfreeze ทุก layer
        - Train ทั้งโมเดลพร้อมกัน
        - Learning rate = 1e-5 (ต่ำมากเพื่อไม่ destroy pretrained weights)
        - 20 epochs + EarlyStopping
        - ได้ accuracy สูงสุด
        """)

    st.header("5. ผลลัพธ์")
    st.success("**Val Accuracy: 80.23%** — สูงกว่า ML Ensemble (34.13%) มากกว่า 2 เท่า")

    st.header("6. แหล่งอ้างอิง")
    st.markdown("""
    - Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML 2019*
    - TensorFlow Documentation: https://www.tensorflow.org
    - Keras Applications: https://keras.io/api/applications/efficientnet
    - Kaggle Dataset: Sports Balls Multiclass Image Classification
    """)

# ======================================================
# หน้า 3: ทดสอบ ML Model
# ======================================================
elif page == "🧪 ทดสอบ ML Model":
    st.title("🧪 ทดสอบ ML Ensemble Model")
    st.markdown("---")

    try:
        ensemble, scaler, idx_to_class = load_ml_models()
        st.success("โหลดโมเดลสำเร็จ ✅")
    except Exception as e:
        st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
        st.stop()

    st.markdown("### อัปโหลดรูปกีฬา แล้วโมเดลจะทำนายประเภทให้")
    st.caption("รองรับไฟล์: JPG, JPEG, PNG")

    uploaded = st.file_uploader("เลือกรูปภาพ", type=['jpg','jpeg','png'], key='ml_upload')

    if uploaded:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded, caption="รูปที่อัปโหลด", use_container_width=True)

        with col2:
            with st.spinner("กำลังทำนาย..."):
                img_bytes = uploaded.read()
                pred_class, confidence, proba = predict_ml(img_bytes, ensemble, scaler, idx_to_class)

            st.markdown("### ผลการทำนาย")
            st.success(f"**{pred_class.replace('_', ' ').title()}**")
            st.metric("Confidence", f"{confidence*100:.1f}%")

            st.markdown("### Top 5 Predictions")
            top5_idx = np.argsort(proba)[::-1][:5]
            for i, idx in enumerate(top5_idx):
                cls  = idx_to_class.get(idx, str(idx))
                prob = proba[idx]
                st.progress(float(prob), text=f"{i+1}. {cls.replace('_',' ').title()} — {prob*100:.1f}%")

# ======================================================
# หน้า 4: ทดสอบ Neural Network
# ======================================================
elif page == "🧪 ทดสอบ Neural Network":
    st.title("🧪 ทดสอบ Neural Network (EfficientNetB0)")
    st.markdown("---")

    try:
        nn_model, idx_to_class_nn = load_nn_model()
        st.success("โหลดโมเดลสำเร็จ ✅")
    except Exception as e:
        st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
        st.stop()

    st.markdown("### อัปโหลดรูปกีฬา แล้วโมเดลจะทำนายประเภทให้")
    st.caption("รองรับไฟล์: JPG, JPEG, PNG")

    uploaded = st.file_uploader("เลือกรูปภาพ", type=['jpg','jpeg','png'], key='nn_upload')

    if uploaded:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded, caption="รูปที่อัปโหลด", use_container_width=True)

        with col2:
            with st.spinner("กำลังทำนาย..."):
                img_bytes = uploaded.read()
                pred_class, confidence, proba = predict_nn(img_bytes, nn_model, idx_to_class_nn)

            st.markdown("### ผลการทำนาย")
            st.success(f"**{pred_class.replace('_', ' ').title()}**")
            st.metric("Confidence", f"{confidence*100:.1f}%")

            st.markdown("### Top 5 Predictions")
            top5_idx = np.argsort(proba)[::-1][:5]
            for i, idx in enumerate(top5_idx):
                cls  = idx_to_class_nn.get(idx, str(idx))
                prob = proba[idx]
                st.progress(float(prob), text=f"{i+1}. {cls.replace('_',' ').title()} — {prob*100:.1f}%")
