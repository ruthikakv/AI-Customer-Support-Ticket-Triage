import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Find the project folder

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load trained AI models

# Load trained AI models or train them if missing

CATEGORY_MODEL_PATH = PROJECT_DIR / "models" / "category_model.joblib"
URGENCY_MODEL_PATH = PROJECT_DIR / "models" / "urgency_model.joblib"

DATA_PATH = PROJECT_DIR / "data" / "tickets.csv"

if CATEGORY_MODEL_PATH.exists() and URGENCY_MODEL_PATH.exists():
    category_model = joblib.load(CATEGORY_MODEL_PATH)
    urgency_model = joblib.load(URGENCY_MODEL_PATH)

else:
    st.info("Preparing AI models for the first time...")

    # Load the training dataset
    df = pd.read_csv(DATA_PATH)

    X = df["ticket_text"]
    y_category = df["category"]
    y_urgency = df["urgency"]

    # Create the category prediction model
    category_model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Create the urgency prediction model
    urgency_model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Train both models using the available dataset
    category_model.fit(X, y_category)
    urgency_model.fit(X, y_urgency)

    st.success("AI models are ready!")

# Support team mapping

support_teams = {
"Billing": "Billing Support Team",
"Account": "Account Support Team",
"Technical": "Technical Support Team",
"Product": "Product Support Team"
}

# Page configuration

st.set_page_config(
page_title="AI Customer Support Ticket Triage",
page_icon="🎫",
layout="centered"
)

# Application title

st.title("🎫 AI Customer Support Ticket Triage")

st.write(
"Enter a customer support ticket below. "
"The AI will predict its category, urgency, "
"and recommended support team."
)

st.divider()

# Ticket input

ticket = st.text_area(
"Customer Ticket",
placeholder="Example: I was charged twice for my subscription...",
height=150
)

# Analyze button

if st.button("Analyze Ticket", type="primary"):


    if not ticket.strip():
        st.warning("Please enter a customer ticket first.")

    else:
        
         # Make predictions
            category = category_model.predict([ticket])[0]
            urgency = urgency_model.predict([ticket])[0]
        
            # Calculate confidence scores
            category_probabilities = category_model.predict_proba([ticket])[0]
            urgency_probabilities = urgency_model.predict_proba([ticket])[0]
        
            category_confidence = max(category_probabilities) * 100
            urgency_confidence = max(urgency_probabilities) * 100
        
            # Recommend support team
            recommended_team = support_teams.get(
                category, "General Support Team"
            )
        
            # Display results
            st.subheader("Analysis Results")
        
            col1, col2 = st.columns(2)
        
            with col1:
                st.metric("Predicted Category", category)
                st.metric("Category Confidence", f"{category_confidence:.1f}%")
        
            with col2:
                st.metric("Predicted Urgency", urgency)
                st.metric("Urgency Confidence", f"{urgency_confidence:.1f}%")
        
            st.info(f"Recommended Team: {recommended_team}")
        
            # Human review warning
            CONFIDENCE_THRESHOLD = 60
        
            if (
                category_confidence < CONFIDENCE_THRESHOLD
                or urgency_confidence < CONFIDENCE_THRESHOLD
            ):
                st.warning(
                    "Human review recommended: the model has low confidence "
                    "in at least one prediction."
                )
            else:
                st.success("No low-confidence warning triggered.")
        
            st.caption(
                "Confidence scores are estimates, not guarantees of correctness. "
                "Review urgent or uncertain tickets before taking action."
            )
        
        