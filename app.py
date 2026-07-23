import streamlit as st
import joblib
import time
import pandas as pd

# -------------------- Page Config --------------------

st.set_page_config(
    page_title="AI Powered Spam Email Detection",
    page_icon="📧",
    layout="wide"
)

# -------------------- Load Model --------------------

@st.cache_resource
def load_models():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# -------------------- Session State --------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- Sidebar --------------------

st.sidebar.title("📊 Project Information")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("""
### Model

- Multinomial Naive Bayes

### Vectorizer

- TF-IDF

### Dataset

SMS Spam Collection

### Language

Python

### Framework

Streamlit
""")

# -------------------- Header --------------------

st.title("📧 AI Powered Spam Email Detection System")

st.markdown("""
Detect whether an Email or SMS is **Spam** or **Ham** using Machine Learning.

Developed using:

- Python
- Scikit-Learn
- TF-IDF
- Multinomial Naive Bayes
""")

st.divider()

# -------------------- Input --------------------

message = st.text_area(
    "Enter Email or SMS",
    height=220,
    placeholder="Congratulations! You have won ₹50,000. Click here to claim..."
)

col1, col2 = st.columns(2)

predict = col1.button("🚀 Predict", use_container_width=True)
clear = col2.button("🗑 Clear", use_container_width=True)

if clear:
    st.rerun()

# -------------------- Prediction --------------------

if predict:

    if message.strip() == "":
        st.warning("Please enter a message.")
        st.stop()

    start = time.time()

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    confidence = max(probability)

    spam_prob = probability[1]

    ham_prob = probability[0]

    end = time.time()

    processing = end - start

    st.divider()

    if prediction == 1:
        st.error("🚨 SPAM MESSAGE DETECTED")
    else:
        st.success("✅ LEGITIMATE (HAM) MESSAGE")

    st.subheader("Confidence")

    st.progress(float(confidence))

    st.write(f"**Confidence:** {confidence*100:.2f}%")

    st.subheader("Prediction Probability")

    prob_df = pd.DataFrame(
        {
            "Class": ["Ham", "Spam"],
            "Probability": [ham_prob, spam_prob],
        }
    )

    st.bar_chart(prob_df.set_index("Class"))

    st.write(f"Processing Time : **{processing:.4f} sec**")

    st.session_state.history.append(
        {
            "Message": message[:60],
            "Prediction": "Spam" if prediction else "Ham",
            "Confidence": f"{confidence*100:.2f}%"
        }
    )

st.divider()

# -------------------- History --------------------

st.subheader("📜 Recent Predictions")

if len(st.session_state.history):

    history = pd.DataFrame(st.session_state.history[::-1])

    st.dataframe(history, use_container_width=True)

else:

    st.info("No predictions yet.")

st.divider()

# -------------------- Architecture --------------------

st.subheader("⚙️ Machine Learning Pipeline")

st.markdown("""User Input
│
▼
Text Cleaning
│
▼
TF-IDF Vectorizer
│
▼
Multinomial Naive Bayes
│
▼
Spam / Ham Prediction
│
▼
Confidence Score
""")

st.divider()

# -------------------- About --------------------

st.subheader("📘 About This Project")

st.markdown("""

### Objective

This project detects Spam Emails and SMS messages using Machine Learning.

---

### Workflow

1. User enters a message.
2. Message is converted into numerical features using TF-IDF.
3. Multinomial Naive Bayes predicts Spam or Ham.
4. Confidence score is displayed.

---

### Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

---

### Machine Learning

- TF-IDF Vectorization
- Multinomial Naive Bayes
- Text Classification
- Natural Language Processing (NLP)

""")

st.divider()

st.caption("AI Powered Spam Email Detection • Week 4 Internship Project")
