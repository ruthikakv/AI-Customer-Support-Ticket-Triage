import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Find project folders

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "tickets.csv"
MODEL_PATH = PROJECT_DIR / "models" / "urgency_model.joblib"

# Load dataset

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Total tickets:", len(df))

# Select input and target

X = df["ticket_text"]
y = df["urgency"]

# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y
)

print("\nTraining tickets:", len(X_train))
print("Testing tickets:", len(X_test))

# Create the AI model

model = Pipeline([
("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
("classifier", LogisticRegression(max_iter=1000))
])

# Train the model

print("\nTraining the urgency prediction model...")
model.fit(X_train, y_train)

print("Urgency model training completed!")

# Test the model

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nUrgency Model Accuracy:")
print(f"{accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save the trained model

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nUrgency model saved successfully!")
print("Model location:", MODEL_PATH)
