import streamlit as st
import joblib
import numpy as np
import json
import google.generativeai as genai
import os
import time
import re
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from io import BytesIO

# --- IMPORTANT: SET THE PATHS FOR TESSERACT AND POPPLER ---
# This line is crucial for the pytesseract library to find the Tesseract OCR engine.
# You MUST change the path below to match where Tesseract is installed on your system.
# For example, on Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
# For Linux, this step is often not needed if tesseract is in your PATH.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Configuration and Setup ---
# Use the provided API key directly for this version.
GEMINI_API_KEY = "AIzaSyDC7OZ-hH8Uy-YVzbu1qOb-D6EltOrV2xk"
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. Please ensure the API key is correct and valid.")
    st.stop()

# --- Load Models and Encoders (using Streamlit's cache) ---
@st.cache_resource
def load_models():
    """Load the trained models and label encoders."""
    try:
        model_crop = joblib.load('model_crop.pkl')
        le_crop = joblib.load('label_encoder_crop.pkl')
        model_fertilizer = joblib.load('model_fertilizer.pkl')
        le_fertilizer = joblib.load('label_encoder_fertilizer.pkl')
        return model_crop, le_crop, model_fertilizer, le_fertilizer
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure `model_crop.pkl`, `label_encoder_crop.pkl`, `model_fertilizer.pkl`, and `label_encoder_fertilizer.pkl` are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading the model files: {e}")
        st.stop()

model_crop, le_crop, model_fertilizer, le_fertilizer = load_models()
llm = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- Helper Functions ---

def predict_crop(N, P, K, temp, humidity, ph, rainfall):
    """Predict the best crop using the trained model."""
    features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    prediction = model_crop.predict(features)
    # Get top 3 predictions
    top_3_indices = np.argsort(model_crop.predict_proba(features)[0])[::-1][:3]
    top_3_crops = le_crop.inverse_transform(top_3_indices)
    return top_3_crops

def predict_fertilizer(N, P, K):
    """Predict the best fertilizer using the trained model."""
    features = np.array([[N, K, P]])  # Note: The order of features is different here
    # Get top 3 predictions
    top_3_indices = np.argsort(model_fertilizer.predict_proba(features)[0])[::-1][:3]
    top_3_fertilizers = le_fertilizer.inverse_transform(top_3_indices)
    return top_3_fertilizers

def get_gemini_recommendation(user_data, crops, fertilizers):
    """Call Gemini API for a detailed recommendation and analysis."""
    crop_list_str = ', '.join(crops)
    fertilizer_list_str = ', '.join(fertilizers)
    
    prompt = f"""
    You are an expert agricultural advisor. A farmer has provided the following soil and weather data and our
    local machine learning models have made some initial predictions.
    
    User Input Data:
    - Nitrogen (N): {user_data['N']}
    - Phosphorus (P): {user_data['P']}
    - Potassium (K): {user_data['K']}
    - Temperature: {user_data['temperature']} °C
    - Humidity: {user_data['humidity']} %
    - pH: {user_data['ph']}
    - Rainfall: {user_data['rainfall']} mm
    
    ML Model Predictions:
    - Recommended Crops: {crop_list_str}
    - Recommended Fertilizers: {fertilizer_list_str}
    
    Based on this information, provide a detailed and comprehensive analysis and recommendation.
    Your response must be a JSON object with the following structure:
    {{
      "overall_summary": "string",
      "crop_recommendations": [
        {{
          "name": "string",
          "rationale": "string",
          "sowing_details": "string",
          "best_practices": "array of strings",
          "harvesting_details": "string"
        }}
      ],
      "fertilizer_recommendations": [
        {{
          "name": "string",
          "rationale": "string",
          "type": "string",
          "dosage_details": "string",
          "application_schedule": "string",
          "how_to_use": "string"
        }}
      ]
    }}
    
    Ensure the `name` field in each recommendation matches one of the ML model's predictions.
    Provide detailed information for each field as an expert would. The `overall_summary` should be a friendly,
    encouraging paragraph summarizing the findings.
    Do not include any extra text or markdown outside of the JSON block.
    """

    try:
        response = llm.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"Failed to get a response from Gemini: {e}"}

def ocr_and_extract_data(file_bytes):
    """Extracts N, P, K, and pH from a soil report using OCR."""
    try:
        # Check if the file is a PDF
        if file_bytes.type == "application/pdf":
            # Pass the poppler_path argument to convert_from_bytes for Windows.
            # You MUST change this path to match your Poppler installation's "bin" folder.
            # For example, on Windows: poppler_path = r'C:\poppler-23.08.0\Library\bin'
            poppler_path = r'C:\Users\PRACHI DHAGE\poppler-0.68.0\bin'
            pages = convert_from_bytes(file_bytes.getvalue(), poppler_path=poppler_path)
            text = "\n".join(pytesseract.image_to_string(page) for page in pages)
        else:
            # Assume it's an image
            image = Image.open(BytesIO(file_bytes.getvalue()))
            text = pytesseract.image_to_string(image)

        # Regex patterns to find numerical values for N, P, K, and pH
        n_match = re.search(r'Nitrogen\s*[:=]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
        p_match = re.search(r'Phosphorus\s*[:=]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
        k_match = re.search(r'Potassium\s*[:=]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
        ph_match = re.search(r'pH\s*[:=]?\s*(\d+\.?\d*)', text, re.IGNORECASE)

        extracted_data = {
            'N': float(n_match.group(1)) if n_match else None,
            'P': float(p_match.group(1)) if p_match else None,
            'K': float(k_match.group(1)) if k_match else None,
            'ph': float(ph_match.group(1)) if ph_match else None
        }

        return extracted_data
    except Exception as e:
        st.error(f"Error during OCR: {e}. Please ensure you have Tesseract installed and the file is a clear image or PDF.")
        return None

# --- Streamlit UI ---
st.set_page_config(
    page_title="AI Crop & Fertilizer Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a sleek, modern UI
st.markdown("""
<style>
    .st-emotion-cache-1g6x58t {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-10qj09x {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        padding: 24px;
        background-color: #ffffff;
    }
    .st-emotion-cache-163mkh {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .st-emotion-cache-4z1l7f {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    .st-emotion-cache-4z1l7f:hover {
        background-color: #45a049;
        color: white;
    }
    .main-header {
        color: #4CAF50;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
        font-weight: bold;
    }
    .subheader {
        color: #555;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }
    .stSpinner > div > div {
        border-top-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Main Page Header
st.markdown("<h1 class='main-header'>🌱 AI-Powered Farm Advisor</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='subheader'>Get personalized crop and fertilizer recommendations based on your soil and weather conditions.</h4>", unsafe_allow_html=True)

# --- Input Method Selection ---
st.markdown("---")
input_method = st.radio("Choose Input Method:", ("Upload Soil Report", "Manual Entry"))
st.markdown("---")

# --- Form State Management ---
if 'form_data' not in st.session_state:
    st.session_state['form_data'] = {
        'N': 70.0, 'P': 70.0, 'K': 70.0,
        'temperature': 25.0, 'humidity': 65.0, 'ph': 6.5,
        'rainfall': 150.0
    }

# --- OCR-based Input Form ---
if input_method == "Upload Soil Report":
    st.header("Upload Soil Report for Auto-fill")
    uploaded_file = st.file_uploader("Upload a soil report (Image or PDF)", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file:
        with st.spinner("Analyzing soil report..."):
            extracted_data = ocr_and_extract_data(uploaded_file)
            if extracted_data:
                st.success("Analysis complete! Review and adjust the values below.")
                for key, value in extracted_data.items():
                    if value is not None:
                        st.session_state['form_data'][key] = value

    st.warning("Please ensure your report has clear text for optimal results. You can manually adjust values below.")

# --- Manual Input Fields ---
st.header("Enter Your Farm Conditions Manually")
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state['form_data']['N'] = st.number_input("Nitrogen (N) Content (kg/ha)", min_value=0.0, max_value=140.0, value=st.session_state['form_data']['N'], step=0.1)
        st.session_state['form_data']['P'] = st.number_input("Phosphorus (P) Content (kg/ha)", min_value=5.0, max_value=145.0, value=st.session_state['form_data']['P'], step=0.1)
        st.session_state['form_data']['K'] = st.number_input("Potassium (K) Content (kg/ha)", min_value=5.0, max_value=205.0, value=st.session_state['form_data']['K'], step=0.1)
    
    with col2:
        st.session_state['form_data']['temperature'] = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=st.session_state['form_data']['temperature'], step=0.1)
        st.session_state['form_data']['humidity'] = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=st.session_state['form_data']['humidity'], step=0.1)
        st.session_state['form_data']['ph'] = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=st.session_state['form_data']['ph'], step=0.1)

    with col3:
        st.session_state['form_data']['rainfall'] = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=st.session_state['form_data']['rainfall'], step=0.1)

    st.markdown("---")
    
    if st.button("Get AI-powered Recommendation", use_container_width=True):
        user_data = st.session_state['form_data']
        
        # --- Run Prediction and Call Gemini ---
        with st.spinner("Analyzing data and generating recommendation..."):
            start_time = time.time()
            
            # Step 1: Get predictions from local ML models
            predicted_crops = predict_crop(user_data['N'], user_data['P'], user_data['K'], user_data['temperature'], user_data['humidity'], user_data['ph'], user_data['rainfall'])
            predicted_fertilizers = predict_fertilizer(user_data['N'], user_data['K'], user_data['P'])

            # Step 2: Get detailed recommendation from Gemini
            gemini_response = get_gemini_recommendation(user_data, predicted_crops, predicted_fertilizers)
            end_time = time.time()

        # --- Display Results ---
        if "error" in gemini_response:
            st.error(gemini_response["error"])
        else:
            st.markdown("<h2 class='main-header'>✅ Your Personalized Farm Plan</h2>", unsafe_allow_html=True)
            
            st.markdown(f"<p class='subheader'><i>Generated in {end_time - start_time:.2f} seconds.</i></p>", unsafe_allow_html=True)

            summary = gemini_response.get("overall_summary", "Summary not available.")
            st.info(summary)

            st.markdown("---")
            
            # Crop Recommendation Section
            st.subheader("🌾 Top Crop Recommendations")
            crop_recs = gemini_response.get("crop_recommendations", [])
            if not crop_recs:
                st.warning("No crop recommendations available from the AI model.")
            for crop_rec in crop_recs:
                with st.expander(f"**{crop_rec.get('name', 'N/A')}**"):
                    st.markdown(f"**Rationale:** {crop_rec.get('rationale', 'N/A')}")
                    st.markdown(f"**Sowing Details:** {crop_rec.get('sowing_details', 'N/A')}")
                    st.markdown(f"**Best Practices:**")
                    best_practices = crop_rec.get("best_practices", [])
                    for bp in best_practices:
                        st.markdown(f"- {bp}")
                    st.markdown(f"**Harvesting Details:** {crop_rec.get('harvesting_details', 'N/A')}")

            st.markdown("---")
            
            # Fertilizer Recommendation Section
            st.subheader("🧪 Top Fertilizer Recommendations")
            fert_recs = gemini_response.get("fertilizer_recommendations", [])
            if not fert_recs:
                st.warning("No fertilizer recommendations available from the AI model.")
            for fert_rec in fert_recs:
                with st.expander(f"**{fert_rec.get('name', 'N/A')}**"):
                    st.markdown(f"**Rationale:** {fert_rec.get('rationale', 'N/A')}")
                    st.markdown(f"**Type:** {fert_rec.get('type', 'N/A')}")
                    st.markdown(f"**Dosage Details:** {fert_rec.get('dosage_details', 'N/A')}")
                    st.markdown(f"**Application Schedule:** {fert_rec.get('application_schedule', 'N/A')}")
                    st.markdown(f"**How to Use:** {fert_rec.get('how_to_use', 'N/A')}")

# --- Footer ---
st.markdown("---")
st.markdown("Built with❤️ using Streamlit, Scikit-learn, and the Gemini API.")
