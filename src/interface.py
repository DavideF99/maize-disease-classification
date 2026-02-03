import streamlit as st
import requests
from PIL import Image
import io

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

# --- STYLING & HEADER ---
st.title("🌽 MaizePath South Africa")
st.markdown("""
**AI-Powered Hierarchical Ensemble for Leaf Pathology:** *A simulated real-world diagnostic tool for South African agricultural resilience.*
---
""")

# --- SIDEBAR: SYSTEM SETTINGS & CONTEXT ---
with st.sidebar:
    st.header("📍 Deployment Context")
    province = st.selectbox("Current Region", list(SA_REGIONS.keys()))
    
    st.divider()
    
    st.header("⚙️ Engine Settings")
    api_url = st.text_input("Backend API Endpoint", "http://localhost:8000/predict")
    
    st.divider()
    
    st.info(f"**Regional Profile:** \n{SA_REGIONS[province]['risk']}")
    st.write(f"**Standard Advice:** \n{SA_REGIONS[province]['advice']}")

# --- MAIN INTERFACE: UPLOAD & ANALYSIS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Sample")
    uploaded_file = st.file_uploader("Upload a maize leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Specimen", use_container_width=True)

with col2:
    st.subheader("🔬 Diagnostic Results")
    
    if uploaded_file is not None:
        if st.button("Run Full-Scale Analysis", type="primary"):
            with st.spinner('Querying Dockerized Ensemble...'):
                # Convert image to bytes
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                
                try:
                    # Request to FastAPI
                    files = {"file": (uploaded_file.name, byte_im, "image/jpeg")}
                    response = requests.post(api_url, files=files, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display Logic
                        if data['status'] == "Diseased":
                            st.error(f"### Result: {data['status']}")
                            
                            # List individual diseases
                            for item in data['diagnoses']:
                                st.write(f"✅ **{item['label']}** Detected (Conf: {item['probability']:.2%})")
                            
                            # --- South African Intelligence Layer ---
                            st.divider()
                            st.warning(f"### 🇿🇦 Actionable Insight for {province}")
                            st.write(SA_REGIONS[province]['advice'])
                            
                            # Specific high-value warning
                            detected_labels = [d['label'] for d in data['diagnoses']]
                            if "GLS" in detected_labels and province == "KwaZulu-Natal":
                                st.info("💡 **Special Note:** In KZN, GLS outbreaks can spread rapidly. Consult the ARC (Agricultural Research Council) immediately for localized spray schedules.")
                        
                        else:
                            st.success(f"### Result: {data['status']}")
                            st.write("The specimen shows no significant signs of foliar disease. Continue regular scouting.")
                            
                        # Technical Metadata
                        with st.expander("See Technical Trace"):
                            st.json(data)
                            
                    else:
                        st.error(f"Server Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Connection Failed: Is the Docker container running at {api_url}?")
    else:
        st.info("Upload an image on the left to begin the diagnostic trace.")

# --- FOOTER ---
st.divider()
st.caption("Developed as a Portfolio Technical Demonstration | Engine: FastAPI + PyTorch | Frontend: Streamlit")