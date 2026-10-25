
import pandas as pd
from pathlib import Path

# Project paths
DATA_FILE = Path("data/tickets.csv")

# Additional realistic customer support tickets.
# Each ticket contains: ticket_text, category, urgency

new_tickets = [
    # BILLING
    ("I was billed for a subscription I cancelled", "Billing", "High"),
    ("Please send me a copy of my latest invoice", "Billing", "Low"),
    ("My refund has not arrived yet", "Billing", "Medium"),
    ("The amount charged is different from the displayed price", "Billing", "High"),
    ("I do not understand the extra fee on my bill", "Billing", "Medium"),
    ("My card was charged but the order failed", "Billing", "High"),
    ("How can I download previous payment receipts?", "Billing", "Low"),
    ("I need help updating my billing address", "Billing", "Low"),
    ("The monthly subscription price increased unexpectedly", "Billing", "Medium"),
    ("I received two invoices for the same purchase", "Billing", "High"),
    ("My refund request is still being processed", "Billing", "Medium"),
    ("I cannot add my payment method", "Billing", "Medium"),
    ("The payment page keeps rejecting my card", "Billing", "Medium"),
    ("I was charged after my free trial ended", "Billing", "High"),
    ("Can you explain the tax amount on my invoice?", "Billing", "Low"),
    ("My payment is showing as pending", "Billing", "Medium"),
    ("I need to correct the name on my invoice", "Billing", "Low"),
    ("The discount was not applied to my order", "Billing", "Medium"),
    ("I paid successfully but still see an unpaid balance", "Billing", "High"),
    ("Please help me cancel an accidental payment", "Billing", "High"),

    # ACCOUNT
    ("I cannot reset my password", "Account", "Medium"),
    ("My account is locked after several login attempts", "Account", "High"),
    ("How do I change the phone number on my profile?", "Account", "Low"),
    ("I am not receiving the email verification code", "Account", "Medium"),
    ("I want to deactivate my account", "Account", "Low"),
    ("My profile changes are not being saved", "Account", "Medium"),
    ("I think someone accessed my account without permission", "Account", "High"),
    ("I cannot sign in even though my password is correct", "Account", "High"),
    ("How can I update my personal details?", "Account", "Low"),
    ("The verification link has expired", "Account", "Medium"),
    ("I no longer have access to my registered email", "Account", "Medium"),
    ("My account settings page is not loading", "Account", "Medium"),
    ("Please help me recover my account", "Account", "Medium"),
    ("I accidentally created two accounts", "Account", "Low"),
    ("My profile picture will not upload", "Account", "Low"),
    ("The login page keeps sending me back to sign in", "Account", "High"),
    ("I need to change my username", "Account", "Low"),
    ("Two-factor authentication is not working", "Account", "High"),
    ("I am unable to verify my phone number", "Account", "Medium"),
    ("How can I update my account password?", "Account", "Low"),

    # TECHNICAL
    ("The website becomes very slow when I search", "Technical", "Medium"),
    ("The mobile application closes unexpectedly", "Technical", "High"),
    ("I get an error whenever I submit the form", "Technical", "High"),
    ("The dashboard is blank after I log in", "Technical", "High"),
    ("The search function is returning no results", "Technical", "Medium"),
    ("The application is not loading on my phone", "Technical", "High"),
    ("I cannot upload a file to the website", "Technical", "Medium"),
    ("The page displays a server error every time", "Technical", "High"),
    ("Notifications are not appearing in the application", "Technical", "Medium"),
    ("The website layout is broken on my tablet", "Technical", "Low"),
    ("The application freezes when I open settings", "Technical", "High"),
    ("I am unable to download a report", "Technical", "Medium"),
    ("The page keeps refreshing by itself", "Technical", "Medium"),
    ("The application is not syncing my latest changes", "Technical", "Medium"),
    ("I get a timeout error when saving information", "Technical", "High"),
    ("The website is unavailable right now", "Technical", "High"),
    ("The app crashes when I try to upload a photo", "Technical", "High"),
    ("The menu buttons do not respond when clicked", "Technical", "Medium"),
    ("I cannot connect to the service using my browser", "Technical", "High"),
    ("The application takes too long to open", "Technical", "Medium"),

    # PRODUCT
    ("Where can I find the product user guide?", "Product", "Low"),
    ("Does this product support wireless connectivity?", "Product", "Low"),
    ("I received the wrong product variant", "Product", "Medium"),
    ("The product specifications are missing from the page", "Product", "Low"),
    ("How do I register my product warranty?", "Product", "Low"),
    ("The product arrived with a missing accessory", "Product", "Medium"),
    ("I need instructions for installing the product", "Product", "Low"),
    ("Is this product compatible with my device?", "Product", "Low"),
    ("The product description does not match the item received", "Product", "High"),
    ("I would like to know the warranty period", "Product", "Low"),
    ("The replacement item has not arrived", "Product", "Medium"),
    ("Can I exchange this product for another model?", "Product", "Medium"),
    ("The product setup instructions are confusing", "Product", "Low"),
    ("I received a damaged item in my order", "Product", "High"),
    ("Where can I find troubleshooting instructions for this product?", "Product", "Low"),
    ("The product is missing its instruction manual", "Product", "Low"),
    ("I need information about available product accessories", "Product", "Low"),
    ("The replacement product is also not working", "Product", "High"),
    ("Can you explain the differences between these two models?", "Product", "Low"),
    ("I want to check whether this product is still under warranty", "Product", "Low"),
]

# Load the existing dataset
df = pd.read_csv(DATA_FILE)

# Create unique ticket IDs for the new records
start_number = len(df) + 1

additional_df = pd.DataFrame(
    [
        {
            "ticket_id": f"T{i:03d}",
            "ticket_text": ticket_text,
            "category": category,
            "urgency": urgency,
        }
        for i, (ticket_text, category, urgency)
        in enumerate(new_tickets, start=start_number)
    ]
)

# Combine existing and new records
combined_df = pd.concat([df, additional_df], ignore_index=True)

# Remove duplicate messages, if any
combined_df = combined_df.drop_duplicates(
    subset=["ticket_text"],
    keep="first"
)

# Validate the dataset before saving
allowed_categories = {"Billing", "Account", "Technical", "Product"}
allowed_urgencies = {"Low", "Medium", "High"}

if combined_df.isnull().any().any():
    raise ValueError("Dataset contains missing values.")

if not set(combined_df["category"]).issubset(allowed_categories):
    raise ValueError("Unexpected category found.")

if not set(combined_df["urgency"]).issubset(allowed_urgencies):
    raise ValueError("Unexpected urgency label found.")

# Save the expanded dataset
combined_df.to_csv(DATA_FILE, index=False)

print("Dataset expansion completed!")
print("Total tickets:", len(combined_df))
print("\nCategory distribution:")
print(combined_df["category"].value_counts())

print("\nUrgency distribution:")
print(combined_df["urgency"].value_counts())

print("\nMissing values:")
print(combined_df.isnull().sum())

