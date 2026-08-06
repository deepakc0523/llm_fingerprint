import sys
import os
import time
import pandas as pd
import streamlit as st

# Add project root to sys.path to ensure module imports work seamlessly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.predictor import LLMFingerprintPredictor

st.set_page_config(
    page_title="LLM Fingerprinting System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for rich styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2563EB;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    """Cache Predictor instance across app reruns."""
    return LLMFingerprintPredictor()

st.markdown('<div class="main-title">LLM Fingerprinting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Identify the originating Large Language Model of AI-generated text using Stylometric, N-Gram & Semantic Feature Fusion</div>', unsafe_allow_html=True)

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading model pipeline: {str(e)}")
    st.stop()

st.markdown("### Input Text")
user_input = st.text_area(
    "Paste AI Generated Text:",
    height=200,
    placeholder="Paste text here to detect originating LLM (e.g. Gemma, Llama, Mistral, Phi, Qwen)..."
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    if not user_input or not user_input.strip():
        st.warning("⚠️ Please paste non-empty AI generated text to analyze.")
    else:
        with st.spinner("Executing fingerprint pipeline & stacked ensemble..."):
            try:
                res = predictor.predict(user_input)
                
                st.markdown("---")
                st.markdown("## Prediction Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">Predicted Model</div>
                        <div class="metric-val">{res['predicted_model']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">Confidence</div>
                        <div class="metric-val">{res['confidence']:.2f} %</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">Inference Time</div>
                        <div class="metric-val">{res['processing_time']:.2f} sec</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                
                # Probability Distribution Chart
                st.markdown("### Model Probability Breakdown")
                
                probs = res["probabilities"]
                df_probs = pd.DataFrame({
                    "Model": list(probs.keys()),
                    "Probability (%)": list(probs.values())
                })
                
                st.bar_chart(df_probs.set_index("Model"))

                # Textual Summary Breakdown
                st.markdown("### Prediction Summary")
                
                summary_lines = ["```"]
                summary_lines.append("====================================")
                summary_lines.append("Predicted Model")
                summary_lines.append(f"{res['predicted_model']}")
                summary_lines.append("\nConfidence")
                summary_lines.append(f"{res['confidence']:.2f} %")
                summary_lines.append("------------------------------------")
                
                for model_name, prob in probs.items():
                    summary_lines.append(f"{model_name:<10} {prob:.1f} %")
                    
                summary_lines.append("------------------------------------")
                summary_lines.append(f"Inference Time : {res['processing_time']:.2f} sec")
                summary_lines.append("====================================")
                summary_lines.append("```")
                
                st.markdown("\n".join(summary_lines))
                
            except Exception as ex:
                st.error(f"Inference Failure: {str(ex)}")
