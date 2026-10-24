import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
from pathlib import Path


# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "tickets.csv"
MODEL_PATH = PROJECT_DIR / "models" / "category_model.joblib"


# 2. Load the dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Total tickets:", len(df))


# 3. Select input and target columns
X = df["ticket_text"]
y = df["category"]


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining tickets:", len(X_train))
print("Testing tickets:", len(X_test))


# 5. Create the machine-learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])


# 6. Train the model
print("\nTraining the category prediction model...")

model.fit(X_train, y_train)

print("Model training completed!")


# 7. Make predictions on unseen test data
y_pred = model.predict(X_test)


# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# 9. Display the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation="vertical"
)

plt.title("Customer Ticket Category Confusion Matrix")
plt.tight_layout()

# Save the confusion matrix as an image
plt.savefig(PROJECT_DIR / "models" / "confusion_matrix.png")
plt.close()

# Save the trained model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nTrained model saved successfully!")
print("Model location:", MODEL_PATH)
print("Confusion matrix saved successfully!")