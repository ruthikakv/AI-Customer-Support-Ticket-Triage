import streamlit as st
import joblib
import pandas as pd
from database import initialize_database, save_ticket, get_all_tickets
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="TicketFlow | AI Support",
    page_icon="🎫",
    layout="wide"
)

# Initialize SQLite database
initialize_database()
# --------------------------------------------------
# 2. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

CATEGORY_MODEL_PATH = PROJECT_DIR / "models" / "category_model.joblib"
URGENCY_MODEL_PATH = PROJECT_DIR / "models" / "urgency_model.joblib"
DATA_PATH = PROJECT_DIR / "data" / "tickets.csv"


# --------------------------------------------------
# 3. LOAD AI MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    if CATEGORY_MODEL_PATH.exists() and URGENCY_MODEL_PATH.exists():

        category_model = joblib.load(CATEGORY_MODEL_PATH)
        urgency_model = joblib.load(URGENCY_MODEL_PATH)

    else:

        df = pd.read_csv(DATA_PATH)

        X = df["ticket_text"]
        y_category = df["category"]
        y_urgency = df["urgency"]

        category_model = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000))
        ])

        urgency_model = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000))
        ])

        category_model.fit(X, y_category)
        urgency_model.fit(X, y_urgency)

    return category_model, urgency_model


category_model, urgency_model = load_models()


# --------------------------------------------------
# 4. SUPPORT TEAM MAPPING
# --------------------------------------------------

support_teams = {
    "Billing": "Billing Support Team",
    "Account": "Account Support Team",
    "Technical": "Technical Support Team",
    "Product": "Product Support Team"
}


# --------------------------------------------------
# 5. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F5F7FB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    background: linear-gradient(120deg, #172554, #2563EB);
    padding: 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 28px;
}

.hero h1 {
    color: white;
    font-size: 32px;
    margin-bottom: 8px;
}

.hero p {
    color: #DBEAFE;
    font-size: 16px;
    margin-bottom: 0;
}

.section-title {
    color: #172554;
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: white;
    border: 1px solid #E5EAF2;
    border-radius: 15px;
    padding: 18px;
}

div.stButton > button {
    background: #2563EB;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 12px 20px;
    font-weight: 600;
    width: 100%;
}

div.stButton > button:hover {
    background: #1D4ED8;
    color: white;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5EAF2;
    padding: 18px;
    border-radius: 12px;
}

textarea {
    border-radius: 10px !important;
}

.footer {
    text-align: center;
    color: #64748B;
    font-size: 13px;
    padding-top: 35px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TICKET HISTORY STORAGE
# --------------------------------------------------

if "ticket_history" not in st.session_state:
    st.session_state.ticket_history = []

# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🎫 TicketFlow")

    st.caption("AI Customer Support Platform")

    st.divider()

    st.markdown("### Navigation")

    selected_page = st.radio(
    "Navigation",
    ["Dashboard", "Analyze Ticket", "Ticket History"],
    label_visibility="collapsed"
)

    st.divider()

    st.markdown("### AI System Status")

    st.success("Models loaded")

    st.caption("Powered by Machine Learning")


# --------------------------------------------------
# 7. HEADER
# --------------------------------------------------

st.markdown(
    '<div class="hero">'
    '<h1>AI Customer Support Ticket Triage</h1>'
    '<p>Intelligent ticket classification and urgency prediction '
    'to help support teams respond efficiently.</p>'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# 8. DASHBOARD PAGE
# --------------------------------------------------

if selected_page == "Dashboard":

    st.markdown(
        '<div class="section-title">Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Monitor analyzed customer support tickets and their distribution."
    )

    # Load tickets from SQLite database
    df = get_all_tickets()

    if not df.empty:

        # Rename database columns
        df = df.rename(columns={
            "ticket": "Ticket",
            "category": "Category",
            "urgency": "Urgency",
            "support_team": "Support Team",
            "category_confidence": "Category Confidence",
            "urgency_confidence": "Urgency Confidence",
            "created_at": "Date Analyzed"
        })

        total_tickets = len(df)

        high_tickets = (df["Urgency"] == "High").sum()
        medium_tickets = (df["Urgency"] == "Medium").sum()
        low_tickets = (df["Urgency"] == "Low").sum()

        # Summary cards
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Tickets", total_tickets)
        col2.metric("🔴 High Priority", int(high_tickets))
        col3.metric("🟡 Medium Priority", int(medium_tickets))
        col4.metric("🟢 Low Priority", int(low_tickets))

        st.divider()

        # Category analytics
        st.markdown("### Tickets by Category")

        category_counts = df["Category"].value_counts()

        # Urgency analytics
        st.markdown("### Tickets by Urgency")

        urgency_counts = df["Urgency"].value_counts()

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.write("Category Distribution")
            st.bar_chart(category_counts)

        with chart_col2:
            st.write("Urgency Distribution")
            st.bar_chart(urgency_counts)

        st.divider()

        # Recent tickets
        st.markdown("### Recent Analyzed Tickets")

        st.dataframe(
            df.head(5),
            width="stretch",
            hide_index=True
        )

    else:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Tickets", 0)
        col2.metric("🔴 High Priority", 0)
        col3.metric("🟡 Medium Priority", 0)
        col4.metric("🟢 Low Priority", 0)

        st.info(
            "No analyzed tickets yet. "
            "Analyze your first ticket to see dashboard analytics."
        )
# --------------------------------------------------
# 9. ANALYZE TICKET PAGE
# --------------------------------------------------

elif selected_page == "Analyze Ticket":

    st.markdown(
        '<div class="section-title">Analyze a Customer Ticket</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter a customer complaint or support request. "
        "The AI will classify it and recommend a support team."
    )

    with st.container(border=True):

        with st.form("ticket_analysis_form"):

            ticket = st.text_area(
                "Customer Ticket",
                placeholder=(
                    "Example: I was charged twice for my subscription..."
                ),
                height=160
            )

            submitted = st.form_submit_button(
                "✨ Analyze Ticket"
            )

    if submitted:

        if not ticket.strip():

            st.warning("Please enter a customer ticket first.")

        else:

            # Make predictions
            category = category_model.predict([ticket])[0]
            urgency = urgency_model.predict([ticket])[0]

            # Calculate confidence
            category_probabilities = category_model.predict_proba([ticket])[0]
            urgency_probabilities = urgency_model.predict_proba([ticket])[0]

            category_confidence = max(category_probabilities) * 100
            urgency_confidence = max(urgency_probabilities) * 100

            # Recommend support team
            recommended_team = support_teams.get(
                category,
                "General Support Team"
            )
            # Save ticket permanently in SQLite database
            save_ticket(
                ticket,
                category,
                urgency,
                recommended_team,
                category_confidence,
                urgency_confidence
            )
            # Save ticket to history
            ticket_record = {
                "Ticket": ticket,
                "Category": category,
                "Urgency": urgency,
                "Support Team": recommended_team,
                "Category Confidence": round(category_confidence, 1),
                "Urgency Confidence": round(urgency_confidence, 1)
            }

            st.session_state.ticket_history.append(ticket_record)

            st.markdown(
                '<div class="section-title">Analysis Results</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Predicted Category", category)

            col2.metric("Predicted Urgency", urgency)

            col3.metric("Category Confidence", f"{category_confidence:.1f}%")

            with st.container(border=True):

                st.markdown("### Recommended Support Team")

                st.success(f"👥 {recommended_team}")

                st.markdown("### Prediction Confidence")

                confidence_col1, confidence_col2 = st.columns(2)

                with confidence_col1:

                    st.write("Category Confidence")

                    st.progress(
                        int(category_confidence)
                    )

                    st.caption(f"{category_confidence:.1f}%")

                with confidence_col2:

                    st.write("Urgency Confidence")

                    st.progress(
                        int(urgency_confidence)
                    )

                    st.caption(f"{urgency_confidence:.1f}%")

            # Human review warning
            CONFIDENCE_THRESHOLD = 60

            if (
                category_confidence < CONFIDENCE_THRESHOLD
                or urgency_confidence < CONFIDENCE_THRESHOLD
            ):

                st.warning(
                    "Human review recommended: "
                    "The model has low confidence in at least one prediction."
                )

            else:

                st.success("No low-confidence warning triggered.")

            st.caption(
                "Confidence scores are estimates, not guarantees of correctness. "
                "Review urgent or uncertain tickets before taking action."
            )
# --------------------------------------------------
# 10. TICKET HISTORY PAGE
# --------------------------------------------------

elif selected_page == "Ticket History":

    st.markdown(
        '<div class="section-title">Ticket History</div>',
        unsafe_allow_html=True
    )

    st.write("Search and filter previously analyzed tickets.")

    history_df = get_all_tickets()

    if not history_df.empty:

        history_df = history_df.rename(columns={
            "ticket": "Ticket",
            "category": "Category",
            "urgency": "Urgency",
            "support_team": "Support Team",
            "category_confidence": "Category Confidence",
            "urgency_confidence": "Urgency Confidence",
            "created_at": "Date Analyzed"
        })

        st.metric("Total Analyzed Tickets", len(history_df))

        search_text = st.text_input(
            "🔍 Search Tickets",
            placeholder="Search using ticket description..."
        )

        col1, col2 = st.columns(2)

        with col1:
            category_filter = st.selectbox(
                "Filter by Category",
                ["All", "Billing", "Account", "Technical", "Product"]
            )

        with col2:
            urgency_filter = st.selectbox(
                "Filter by Urgency",
                ["All", "High", "Medium", "Low"]
            )

        filtered_df = history_df.copy()

        if search_text:
            filtered_df = filtered_df[
                filtered_df["Ticket"].str.contains(
                    search_text, case=False, na=False
                )
            ]

        if category_filter != "All":
            filtered_df = filtered_df[
                filtered_df["Category"] == category_filter
            ]

        if urgency_filter != "All":
            filtered_df = filtered_df[
                filtered_df["Urgency"] == urgency_filter
            ]

        st.markdown("### Filtered Tickets")
        st.caption(
            f"Showing {len(filtered_df)} of {len(history_df)} tickets"
        )

        if not filtered_df.empty:

            display_columns = [
                "Ticket",
                "Category",
                "Urgency",
                "Support Team",
                "Category Confidence",
                "Urgency Confidence",
                "Date Analyzed"
            ]

            st.dataframe(
                filtered_df[display_columns],
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning("No tickets match your search or filters.")

    else:
        st.info(
            "No ticket history available yet. Analyze a ticket to see it here."
        )
elif selected_page == "Ticket History":

    st.markdown(
        '<div class="section-title">Ticket History</div>',
        unsafe_allow_html=True
    )

    st.write("Search and filter previously analyzed tickets.")

    history_df = get_all_tickets()

    if not history_df.empty:

        history_df = history_df.rename(columns={
            "ticket": "Ticket",
            "category": "Category",
            "urgency": "Urgency",
            "support_team": "Support Team",
            "category_confidence": "Category Confidence",
            "urgency_confidence": "Urgency Confidence",
            "created_at": "Date Analyzed"
        })

        st.metric("Total Analyzed Tickets", len(history_df))

        search_text = st.text_input(
            "🔍 Search Tickets",
            placeholder="Search using ticket description..."
        )

        col1, col2 = st.columns(2)

        with col1:
            category_filter = st.selectbox(
                "Filter by Category",
                ["All", "Billing", "Account", "Technical", "Product"]
            )

        with col2:
            urgency_filter = st.selectbox(
                "Filter by Urgency",
                ["All", "High", "Medium", "Low"]
            )

        filtered_df = history_df.copy()

        if search_text:
            filtered_df = filtered_df[
                filtered_df["Ticket"].str.contains(
                    search_text, case=False, na=False
                )
            ]

        if category_filter != "All":
            filtered_df = filtered_df[
                filtered_df["Category"] == category_filter
            ]

        if urgency_filter != "All":
            filtered_df = filtered_df[
                filtered_df["Urgency"] == urgency_filter
            ]

        st.markdown("### Filtered Tickets")
        st.caption(
            f"Showing {len(filtered_df)} of {len(history_df)} tickets"
        )

        if not filtered_df.empty:

            display_columns = [
                "Ticket",
                "Category",
                "Urgency",
                "Support Team",
                "Category Confidence",
                "Urgency Confidence",
                "Date Analyzed"
            ]

            st.dataframe(
                filtered_df[display_columns],
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning("No tickets match your search or filters.")

    else:
        st.info(
            "No ticket history available yet. Analyze a ticket to see it here."
        )