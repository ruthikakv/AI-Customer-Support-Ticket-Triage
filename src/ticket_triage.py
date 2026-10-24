import joblib
from pathlib import Path

# Find the project directory

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load both trained models

CATEGORY_MODEL_PATH = PROJECT_DIR / "models" / "category_model.joblib"
URGENCY_MODEL_PATH = PROJECT_DIR / "models" / "urgency_model.joblib"

category_model = joblib.load(CATEGORY_MODEL_PATH)
urgency_model = joblib.load(URGENCY_MODEL_PATH)

print("\nAI Customer Support Ticket Triage")
print("---------------------------------")

# Get ticket text from the user

ticket = input("Enter customer ticket: ").strip()

if not ticket:
    print("Please enter a valid customer ticket.")
else:
# Predict category and urgency
    category = category_model.predict([ticket])[0]
    urgency = urgency_model.predict([ticket])[0]


# Display results

# Recommend a support team
support_teams = {
    "Billing": "Billing Support Team",
    "Account": "Account Support Team",
    "Technical": "Technical Support Team",
    "Product": "Product Support Team"
}

recommended_team = support_teams.get(
    category, "General Support Team"
)

# Display results
print("\n----- TICKET ANALYSIS -----")
print("Customer Ticket:", ticket)
print("Predicted Category:", category)
print("Predicted Urgency:", urgency)
print("Recommended Team:", recommended_team)



