import streamlit as st

from utils import clean_text
from model import predict_depression
from explain import explain_text


st.set_page_config(
    page_title="Explainable Depression Detection Chatbot",
    layout="centered"
)

st.title("🧠 Explainable AI Chatbot for Depression Detection")
st.caption(
    "*This tool provides preliminary screening only and is not a medical diagnosis.*"
)

# Text input
user_text = st.text_area(
    "Enter your message",
    placeholder="Type how you have been feeling...",
    height=150
)

# Analyze button
analyze_btn = st.button("Analyze")

if analyze_btn:
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_text)

        prediction, confidence = predict_depression(cleaned)
        explanations = explain_text(cleaned)

        # Outputs
        st.subheader("Results")
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Confidence Score:** {confidence:.2f}")

        # Explanation section
        st.subheader("Important Words Influencing Prediction")

        if explanations:
            for word, value in explanations:
                if abs(value) > 0.05:
                    if value > 0:
                        st.markdown(
                            f"<span style='background-color:#ffcccc;padding:4px;border-radius:4px'>{word}</span>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<span style='background-color:#ccffcc;padding:4px;border-radius:4px'>{word}</span>",
                            unsafe_allow_html=True
                        )
        else:
            st.info("No significant explanatory words found.")

# Footer disclaimer
st.markdown(
    """
    ---
    **Disclaimer:**  
    This chatbot is for educational and research purposes only.
    """
)