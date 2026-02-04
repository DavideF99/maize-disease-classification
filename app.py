import streamlit as st
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import io
from models import GatekeeperModel, MaizeDiseaseModel

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MaizePath SA | Diagnostic Portal",
    page_icon="🌽",
    layout="wide"
)

# --- SOUTH AFRICAN REGIONAL KNOWLEDGE BASE ---
SA_REGIONS = {
    "KwaZulu-Natal": {
        "risk": "High GLS & Gray Leaf Spot Sensitivity",
        "advice": "Humid coastal and mist-belt conditions favor Gray Leaf Spot. Ensure adequate spacing for airflow and consider fungicides during prolonged wet periods."
    },
    "Free State / North West": {
        "risk": "NCLB & Drought Stress Hotspot",
        "advice": "Dryland maize is prone to Northern Corn Leaf Blight after late rains. Focus on residue management to reduce inoculum levels from previous seasons."
    },
    "Mpumalanga / Gauteng": {
        "risk": "Common Rust / Highveld Risk",
        "advice": "Cooler night temperatures favor Common Rust development. Monitor Highveld hybrids early in the season for rust pustules on upper leaves."
    },
    "Eastern Cape": {
        "risk": "Mixed Pathology / Smallholder Focus",
        "advice": "Coastal humidity meets inland heat. Regular scouting is recommended during the silking stage as mixed infections are common here."
    },
    "Limpopo": {
        "risk": "Heat Stress & Sorghum Downy Mildew",
        "advice": "High temperatures can mask early symptoms. Look for chlorotic striping and ensure irrigation timing doesn't create localized high humidity."
    }
}

# --- CONFIG & MODELS ---
DEVICE = "cpu" # Force CPU for Hugging Face free tier
GK_PATH = "checkpoints/gatekeeper/gatekeeper-epoch=11-gatekeeper_val_f1=0.95.ckpt"
HERO_PATH = "checkpoints/hero_model/maize-epoch=19-val_f1=0.66.ckpt"
LABEL_NAMES = ['GLS', 'NCLB', 'PLS', 'CR', 'SR', 'Healthy', 'Other', 'Unidentified']
THRESHOLDS = [0.3614, 0.4449, 0.2601, 0.2555, 0.9848, 0.6011, 0.2134, 0.2373]

@st.cache_resource
def load_models():
    gk = GatekeeperModel.load_from_checkpoint(GK_PATH, map_location=DEVICE).to(DEVICE).eval()
    hero = MaizeDiseaseModel.load_from_checkpoint(HERO_PATH, map_location=DEVICE).to(DEVICE).eval()
    return gk, hero

# Preprocessing pipeline
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# --- STYLING & HEADER ---
st.title("🌽 MaizePath South Africa")
st.markdown("""
**AI-Powered Hierarchical Ensemble for Leaf Pathology:** *A high-precision diagnostic tool for South African agricultural resilience.*
---
""")

# --- SIDEBAR: SYSTEM SETTINGS & CONTEXT ---
with st.sidebar:
    st.header("📍 Deployment Context")
    province = st.selectbox("Current Region", list(SA_REGIONS.keys()))
    
    st.divider()
    
    st.info(f"**Regional Profile:** \n{SA_REGIONS[province]['risk']}")
    st.write(f"**Standard Advice:** \n{SA_REGIONS[province]['advice']}")
    
    st.divider()
    st.caption("Engine: Hierarchical Ensemble (EfficientNet-B0) | Device: CPU")

# --- MAIN INTERFACE: UPLOAD & ANALYSIS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Sample")
    uploaded_file = st.file_uploader("Upload a maize leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Specimen", use_container_width=True)

with col2:
    st.subheader("🔬 Diagnostic Results")
    
    if uploaded_file is not None:
        if st.button("Run Full-Scale Analysis", type="primary"):
            with st.spinner('Running Hierarchical Diagnostic Engine...'):
                try:
                    gk_model, hero_model = load_models()
                    
                    # Preprocess
                    img_np = np.array(image)
                    input_tensor = transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        # --- STAGE 1: Gatekeeper ---
                        gk_prob = torch.sigmoid(gk_model(input_tensor)).item()
                        
                        # --- STAGE 2: Logic Branching ---
                        if gk_prob < 0.3:
                            st.success(f"### Result: Healthy")
                            st.write("The specimen shows no significant signs of foliar disease. Confidence: {:.2%}".format(1 - gk_prob))
                        else:
                            # Specialist Pass
                            hero_probs = torch.sigmoid(hero_model(input_tensor)).squeeze().cpu().numpy()
                            detected = []
                            for i, p in enumerate(hero_probs):
                                if p >= THRESHOLDS[i] and i != 5: # Exclude Healthy index
                                    detected.append({"label": LABEL_NAMES[i], "probability": float(p)})
                            
                            if not detected:
                                st.success(f"### Result: Healthy (Low Symptom)")
                                st.write("Minimal symptoms detected, likely below clinical thresholds. Continue regular scouting.")
                            else:
                                st.error(f"### Result: Diseased")
                                for item in detected:
                                    st.write(f"✅ **{item['label']}** Detected (Conf: {item['probability']:.2%})")
                                
                                # --- South African Intelligence Layer ---
                                st.divider()
                                st.warning(f"### 🇿🇦 Actionable Insight for {province}")
                                st.write(SA_REGIONS[province]['advice'])
                                
                                if "GLS" in [d['label'] for d in detected] and province == "KwaZulu-Natal":
                                    st.info("💡 **Special Note:** In KZN, GLS outbreaks can spread rapidly. Consult the ARC (Agricultural Research Council) immediately for localized spray schedules.")
                                    
                except Exception as e:
                    st.error(f"Analysis Failed: {str(e)}")
    else:
        st.info("Upload an image on the left to begin the diagnostic trace.")

# --- FOOTER ---
st.divider()
st.caption("Developed as a Portfolio Technical Demonstration | Model Card available at root README.md")
