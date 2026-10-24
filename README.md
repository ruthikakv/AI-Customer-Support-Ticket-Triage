# AI-Customer-Support-Ticket-Triage

## Project Overview

AI Customer Support Ticket Triage is a machine learning-based application that automatically analyzes customer support tickets and helps organize them according to their category and urgency.

The application predicts the ticket category, estimates its urgency, recommends a suitable support team, and displays model confidence scores to help identify predictions that may need human review.

## Features

* **Ticket Classification:** Classifies tickets into Billing, Account, Technical, and Product categories.
* **Urgency Prediction:** Predicts Low, Medium, or High urgency.
* **Support Team Recommendation:** Suggests the appropriate team based on the predicted category.
* **Confidence Scores:** Displays the model's estimated confidence for category and urgency predictions.
* **Human Review Alerts:** Flags predictions when either confidence score is below the configured threshold.
* **Interactive Dashboard:** Provides a simple web interface built with Streamlit.
* **Input Validation:** Warns users when they submit an empty ticket.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Joblib
* Streamlit
* Matplotlib

## Project Structure

```text
AI-Customer-Support-Ticket-Triage/
├── data/
│   └── tickets.csv
├── models/
│   ├── category_model.joblib
│   ├── urgency_model.joblib
│   └── confusion_matrix.png
├── src/
│   ├── prepare_data.py
│   ├── train_category_model.py
│   ├── train_urgency_model.py
│   ├── predict.py
│   ├── ticket_triage.py
│   └── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project folder:

```bash
cd AI-Customer-Support-Ticket-Triage
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run src/app.py
```

Open the local address displayed in the terminal to use the application.

## How It Works

1. The user enters a customer support ticket.
2. The category model predicts the ticket category.
3. The urgency model predicts the urgency level.
4. The application recommends a support team.
5. Confidence scores are displayed for both predictions.
6. A human-review warning appears when either score is below the configured threshold.

## Dataset

The current project uses a small synthetic dataset containing 100 labeled customer support tickets.

The dataset is intended for learning and prototyping. The current model results should not be interpreted as proof of reliable performance on real customer support data.

## Future Enhancements

* Expand the dataset with more realistic and diverse ticket examples.
* Evaluate performance using larger, independent test data.
* Add ticket history and a searchable ticket dashboard.
* Store tickets and predictions in a database.
* Add authentication for support staff.
* Improve urgency classification and human-review workflows.
* Deploy the application online.

## Limitations

The model may make incorrect predictions, especially for tickets that differ from the training examples. Confidence scores are estimates and do not guarantee correctness. Human review is important before taking action on urgent or uncertain tickets.

## Author

Developed as a machine learning project to explore automated customer support ticket classification, urgency prediction, and support team recommendation.
