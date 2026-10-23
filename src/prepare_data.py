import pandas as pd

# Load the customer support ticket dataset

df = pd.read_csv("data/tickets.csv")

# Display the first five tickets

print("\nFirst five tickets:")
print(df.head())

# Display the dataset size

print("\nDataset shape:")
print(df.shape)

# Check for missing values

print("\nMissing values:")
print(df.isnull().sum())

# Display category counts

print("\nCategory distribution:")
print(df["category"].value_counts())

# Display urgency counts

print("\nUrgency distribution:")
print(df["urgency"].value_counts())

# Check for duplicate ticket IDs

print("\nDuplicate ticket IDs:", df["ticket_id"].duplicated().sum())
