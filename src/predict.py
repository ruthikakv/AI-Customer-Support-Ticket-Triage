import joblib
from pathlib import Path

# Find the project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Find the saved model
MODEL_PATH = PROJECT_DIR / "models" / "category_model.joblib"

# Load the trained AI model
model = joblib.load(MODEL_PATH)

print("AI Customer Support Ticket Triage")
print("---------------------------------")

# Enter a new customer ticket
ticket = input("Enter customer ticket: ")

# Predict the ticket category
prediction = model.predict([ticket])

# Display the result
print("\nPredicted Category:", prediction[0])