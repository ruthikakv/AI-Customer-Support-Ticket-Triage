import streamlit as st
import joblib
from pathlib import Path

# Find the project folder

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load trained AI models

CATEGORY_MODEL_PATH = PROJECT_DIR / "models" / "category_model.joblib"
URGENCY_MODEL_PATH = PROJECT_DIR / "models" / "urgency_model.joblib"

category_model = joblib.load(CATEGORY_MODEL_PATH)
urgency_model = joblib.load(URGENCY_MODEL_PATH)

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
        
        