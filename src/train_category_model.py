
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt


# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "tickets.csv"
MODEL_PATH = PROJECT_DIR / "models" / "category_model.joblib"
REPORT_PATH = PROJECT_DIR / "models" / "category_evaluation.txt"
MATRIX_PATH = PROJECT_DIR / "models" / "confusion_matrix.png"


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


# 7. Predict categories for unseen test tickets
y_pred = model.predict(X_test)


# 8. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)

recall = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)

f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)


# 9. Display evaluation results
print("\n========== CATEGORY MODEL EVALUATION ==========")

print(f"Accuracy:          {accuracy:.2%}")
print(f"Macro Precision:   {precision:.2%}")
print(f"Macro Recall:      {recall:.2%}")
print(f"Macro F1-score:    {f1:.2%}")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\nDetailed Classification Report:")
print(report)


# 10. Save the detailed report
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("CUSTOMER SUPPORT CATEGORY MODEL EVALUATION\n\n")
    file.write(f"Total tickets: {len(df)}\n")
    file.write(f"Training tickets: {len(X_train)}\n")
    file.write(f"Testing tickets: {len(X_test)}\n\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro Precision: {precision:.4f}\n")
    file.write(f"Macro Recall: {recall:.4f}\n")
    file.write(f"Macro F1-score: {f1:.4f}\n\n")
    file.write("Detailed Classification Report:\n")
    file.write(report)


# 11. Save the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation="vertical"
)

plt.title("Customer Ticket Category Confusion Matrix")
plt.tight_layout()
plt.savefig(MATRIX_PATH)
plt.close()


# 12. Save the trained model
joblib.dump(model, MODEL_PATH)


print("\nTraining and evaluation completed!")
print("Model saved:", MODEL_PATH)
print("Evaluation report saved:", REPORT_PATH)
print("Confusion matrix saved:", MATRIX_PATH)

