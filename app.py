import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.sentiment import train_sentiment_model, analyze_sentiment
from src.topic_classifier import classify_topic
from src.priority import detect_priority
from src.response_generator import generate_response
from src.database import (
    seed_demo_data,
    get_reviews_by_business,
    update_suggested_response,
    approve_review_response
)
from src.customer_form import render_customer_review_form
from src.login import render_login_page, logout_employee
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Page configuration
st.set_page_config(
    page_title="Revio AI | Intelligent Review Analysis & Resolution",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom modern CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background gradient styling */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(30, 27, 75, 0.4) 0%, transparent 40%),
                radial-gradient(circle at 90% 10%, rgba(14, 116, 144, 0.25) 0%, transparent 45%),
                radial-gradient(circle at 50% 80%, rgba(76, 29, 149, 0.25) 0%, transparent 50%),
                #070B14;
    color: #F1F5F9;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0B1120 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

section[data-testid="stSidebar"] div.stRadio > label {
    font-weight: 600;
    color: #94A3B8;
}

/* Modern Card Container */
.revio-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(15, 23, 42, 0.7) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.revio-card:hover {
    transform: translateY(-3px);
    border-color: rgba(139, 92, 246, 0.35);
    box-shadow: 0 20px 30px -10px rgba(99, 102, 241, 0.2), 0 0 15px rgba(139, 92, 246, 0.15);
}

/* Hero Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.8) 0%, rgba(15, 23, 42, 0.9) 50%, rgba(14, 116, 144, 0.4) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 28px;
    box-shadow: 0 20px 35px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #FFFFFF 30%, #C4B5FD 70%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: 8px;
    max-width: 800px;
    line-height: 1.5;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(139, 92, 246, 0.18);
    border: 1px solid rgba(139, 92, 246, 0.35);
    color: #C4B5FD;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(145deg, rgba(20, 27, 45, 0.85) 0%, rgba(13, 18, 30, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
}

.metric-card.total {
    border-left: 4px solid #6366F1;
}
.metric-card.positive {
    border-left: 4px solid #10B981;
}
.metric-card.neutral {
    border-left: 4px solid #F59E0B;
}
.metric-card.negative {
    border-left: 4px solid #EF4444;
}
.metric-card.priority {
    border-left: 4px solid #EC4899;
}

.metric-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.metric-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 8px 0 2px 0;
    line-height: 1.1;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.metric-sub {
    font-size: 0.78rem;
    color: #64748B;
    font-weight: 500;
}

/* Section Header Component */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    margin: 0;
}

.section-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(59, 130, 246, 0.25));
    border: 1px solid rgba(139, 92, 246, 0.3);
    color: #A78BFA;
    font-size: 1rem;
}

/* Priority & Sentiment Badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-high {
    background: rgba(239, 68, 68, 0.18);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.badge-medium {
    background: rgba(245, 158, 11, 0.18);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.badge-low {
    background: rgba(59, 130, 246, 0.18);
    color: #60A5FA;
    border: 1px solid rgba(59, 130, 246, 0.35);
}

.badge-pos {
    background: rgba(16, 185, 129, 0.18);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.badge-neu {
    background: rgba(245, 158, 11, 0.18);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.badge-neg {
    background: rgba(239, 68, 68, 0.18);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

/* Expanders styling */
div[data-testid="stExpander"] {
    background: rgba(17, 24, 39, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    margin-bottom: 12px;
    transition: border-color 0.2s ease;
}

div[data-testid="stExpander"]:hover {
    border-color: rgba(139, 92, 246, 0.35) !important;
}

/* Urgent Alert Box */
.urgent-banner {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 12px;
    color: #FCA5A5;
    font-weight: 500;
}

/* Custom buttons */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.35) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(124, 58, 237, 0.5) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.5) !important;
}

/* Sidebar decorative header */
.sidebar-brand {
    padding: 10px 0 20px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 20px;
}
.sidebar-brand h3 {
    margin: 0;
    background: linear-gradient(135deg, #FFFFFF 0%, #C4B5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 1.4rem;
}
.sidebar-brand p {
    color: #64748B;
    font-size: 0.8rem;
    margin: 4px 0 0 0;
}

/* Streamlit Dataframe Dark styling tweak */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# Reusable Plotly dark navy template styling helper
def apply_revio_plotly_theme(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#CBD5E1", size=12),
        margin=dict(l=30, r=30, t=50, b=40),
        height=height,
        title=dict(
            font=dict(color="#F8FAFC", size=14, family="Plus Jakarta Sans, sans-serif"),
            x=0.03,
            y=0.95
        ),
        legend=dict(
            font=dict(color="#94A3B8"),
            bgcolor="rgba(15, 23, 42, 0.4)",
            bordercolor="rgba(255, 255, 255, 0.05)",
            borderwidth=1
        )
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.1)",
        tickfont=dict(color="#94A3B8")
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.1)",
        tickfont=dict(color="#94A3B8")
    )
    return fig

# Train model once per session
@st.cache_resource
def load_model():
    return train_sentiment_model()

model = load_model()

# Model evaluation dataset
evaluation_reviews = [
    "The product quality is excellent and I love it",
    "Amazing customer support and quick delivery",
    "Very satisfied with my purchase",
    "The product exceeded my expectations",
    "Great service and excellent value",
    "I received my package earlier than expected",
    "Everything works perfectly",
    "The support team resolved my issue quickly",
    "Wonderful experience, highly recommended",
    "I would happily purchase this again",

    "The product is average",
    "Delivery took the expected amount of time",
    "The item works as described",
    "The packaging is standard",
    "The price is reasonable",
    "I received the correct product",
    "The service was acceptable",
    "The product has basic features",
    "Nothing unusual about my experience",
    "The order was processed normally",

    "The product quality is terrible",
    "My delivery was delayed by several days",
    "The customer support team ignored me",
    "I received a broken item",
    "I want my money back",
    "The product stopped working immediately",
    "This was a very disappointing experience",
    "I was charged twice for one purchase",
    "The company did not resolve my complaint",
    "I cannot recommend this service"
]

actual_labels = (
    ["positive"] * 10
    + ["neutral"] * 10
    + ["negative"] * 10
)

predicted_labels = model.predict(evaluation_reviews)

model_cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=["positive", "neutral", "negative"]
)

model_accuracy = accuracy_score(actual_labels, predicted_labels)
model_precision = precision_score(
    actual_labels, predicted_labels, average="macro", zero_division=0
)
model_recall = recall_score(
    actual_labels, predicted_labels, average="macro", zero_division=0
)
model_f1 = f1_score(
    actual_labels, predicted_labels, average="macro", zero_division=0
)

# Analysis function
def analyze_review(review):
    sentiment_result = analyze_sentiment(review, model)
    sentiment = sentiment_result["sentiment"]

    topics = classify_topic(review)

    priority = detect_priority(review, sentiment)

    response = generate_response(
        review,
        sentiment,
        topics,
        priority
    )

    return {
        "Sentiment": sentiment,
        "Confidence": sentiment_result["confidence"],
        "Topic": ", ".join(topics),
        "Priority": priority,
        "Suggested Response": response
    }

# Ensure demo business and employee are seeded
seed_demo_data()

# Page routing: Customer review submission vs. Employee dashboard
# Supports URL query params: ?page=submit_review&biz=<business-id>
query_params = st.query_params
active_page = query_params.get("page", "dashboard")
biz_param = query_params.get("biz", "demo-corp")

if active_page == "submit_review":
    render_customer_review_form(business_id=biz_param, model=model)
    st.stop()

# Protect Employee Dashboard: require authentication in session_state
if not st.session_state.get("authenticated", False):
    render_login_page()
    st.stop()

# Retrieve securely authenticated session variables
auth_username = st.session_state.get("username", "Employee")
auth_business_id = st.session_state.get("business_id", "demo-corp")
auth_business_name = st.session_state.get("business_name", "Business")
auth_role = st.session_state.get("role", "employee")

# Polished Sidebar
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="background: linear-gradient(135deg, #7C3AED, #38BDF8); border-radius:10px; width:34px; height:34px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:18px; color:white; box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);">⚡</div>
            <div>
                <h3>Revio AI</h3>
                <p>Intelligence Platform</p>
            </div>
        </div>
        <div style="margin-top: 14px; padding: 10px 12px; border-radius: 10px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06);">
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Active Workspace</div>
            <div style="font-size: 0.88rem; color: #F8FAFC; font-weight: 700; margin-top: 2px;">{auth_business_name}</div>
            <div style="font-size: 0.75rem; color: #64748B;">User: <span style="color:#A78BFA; font-weight:600;">{auth_username}</span> ({auth_role})</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Sign Out", use_container_width=True):
        logout_employee()

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:#94A3B8; font-weight:700; margin-bottom:6px;'>Dashboard Views</p>", unsafe_allow_html=True)
    view_mode = st.radio(
        "View Mode",
        [
            "⚡ Live Customer Reviews",
            "🔬 Internal AI Testing",
            "📥 CSV Data Import",
            "🧪 Demonstration Dataset"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    df = pd.DataFrame()
    data_source_badge = ""

    if view_mode == "⚡ Live Customer Reviews":
        data_source_badge = "LIVE VERIFIED FEEDBACK"
        db_records = get_reviews_by_business(auth_business_id)
        if db_records:
            df = pd.DataFrame(db_records)
            if "review_text" in df.columns and "review" not in df.columns:
                df["review"] = df["review_text"]
            if "id" in df.columns and "review_id" not in df.columns:
                df["review_id"] = df["id"]
            st.success(f"Connected to live feedback database ({len(df)} customer submissions).")
        else:
            df = pd.DataFrame()

    elif view_mode == "🔬 Internal AI Testing":
        data_source_badge = "INTERNAL MODEL SANDBOX"
        st.markdown("<p style='font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:#A78BFA; font-weight:700; margin-bottom:4px;'>Internal Feedback Tester</p>", unsafe_allow_html=True)
        review_text = st.text_area(
            "Enter customer review",
            placeholder="Type sample text to test sentiment, topics, priority and response generation...",
            height=130,
            label_visibility="collapsed"
        )
        analyze_clicked = st.button("✨ Run AI Inference", use_container_width=True)

    elif view_mode == "📥 CSV Data Import":
        data_source_badge = "EXTERNAL CSV IMPORT"
        st.markdown("<p style='font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:#38BDF8; font-weight:700; margin-bottom:4px;'>Import Historical CSV</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.size > 20 * 1024 * 1024:
                    st.error("Uploaded file exceeds 20MB limit.")
                    df = pd.DataFrame()
                else:
                    df = pd.read_csv(uploaded_file, nrows=5000)
                    if "review" not in df.columns:
                        st.error("Invalid CSV structure: CSV must contain a 'review' column.")
                        df = pd.DataFrame()
                    else:
                        st.info(f"Loaded {len(df)} rows from uploaded file: `{uploaded_file.name}`. Note: Imported reviews remain in local session memory and do not alter your live customer database.")
            except Exception as e:
                st.error(f"Error parsing CSV file: {str(e)}")
                df = pd.DataFrame()

    elif view_mode == "🧪 Demonstration Dataset":
        data_source_badge = "SYNTHETIC DEMO BENCHMARK"
        st.info("Demonstration benchmark data with 9 diverse customer issues across support, delivery, and pricing.")
        load_demo_btn = st.checkbox("Load Demonstration Reviews", value=True)
        if load_demo_btn:
            df = pd.read_csv("data/sample_reviews.csv")
            st.caption("ℹ️ Demo data is isolated for presentation and does not mix with live customer reviews.")
        else:
            df = pd.DataFrame()

    st.markdown(f"""
    <div style='margin-top: 30px; padding: 14px; border-radius: 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);'>
        <div style='font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;'>Active Architecture</div>
        <div style='font-size: 0.8rem; color: #CBD5E1; margin-top: 4px;'>• TF-IDF + Logistic Regression</div>
        <div style='font-size: 0.8rem; color: #CBD5E1;'>• Rule-Based Topic Extraction</div>
        <div style='font-size: 0.8rem; color: #CBD5E1;'>• Heuristic Priority Scorer</div>
    </div>
    <div style='margin-top: 14px; padding: 12px; border-radius: 12px; background: rgba(139,92,246,0.08); border: 1px solid rgba(139,92,246,0.25); text-align: center;'>
        <div style='font-size: 0.72rem; color: #C4B5FD; text-transform: uppercase; font-weight: 700;'>Public Customer Form</div>
        <a href='?page=submit_review&biz={auth_business_id}' target='_blank' style='font-size: 0.82rem; color: #38BDF8; font-weight: 600; text-decoration: none; display: block; margin-top: 4px;'>🔗 Open Review Widget</a>
    </div>
    """, unsafe_allow_html=True)


# Hero Banner Header
st.markdown(f"""
<div class="hero-container">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div class="hero-badge">⚡ AI-Powered Customer Resolution</div>
        <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); padding:4px 12px; border-radius:8px; font-size:0.75rem; color:#94A3B8; font-weight:700; letter-spacing:0.06em;">
            SOURCE: <span style="color:#C4B5FD;">{data_source_badge}</span>
        </div>
    </div>
    <h1 class="hero-title">{auth_business_name} Dashboard</h1>
    <div class="hero-subtitle">
        Deep customer sentiment analytics, automated complaint triage, and intelligent incident response recommendations engineered for modern customer experience teams.
    </div>
</div>
""", unsafe_allow_html=True)

# Single review analysis display (Internal AI Testing mode)
if view_mode == "🔬 Internal AI Testing" and 'analyze_clicked' in locals() and analyze_clicked:
    clean_single_review = review_text.strip() if review_text else ""
    if clean_single_review:
        if len(clean_single_review) > 5000:
            st.error("Review text exceeds the 5,000 character analysis limit.")
        else:
            result = analyze_review(clean_single_review)

            st.markdown("""
            <div class="section-header">
                <span class="section-icon">🔬</span>
                <h3 class="section-title">Internal AI Model Evaluation Sandbox</h3>
            </div>
            """, unsafe_allow_html=True)

            sent_badge_class = "badge-pos" if result["Sentiment"] == "positive" else ("badge-neu" if result["Sentiment"] == "neutral" else "badge-neg")
            prio_badge_class = "badge-high" if result["Priority"] == "High" else ("badge-medium" if result["Priority"] == "Medium" else "badge-low")

            col_rev1, col_rev2 = st.columns([1, 1])

            with col_rev1:
                st.markdown(f"""
                <div class="revio-card">
                    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:#94A3B8; font-weight:700; margin-bottom:8px;">Test Input Text</div>
                    <div style="font-size:1.05rem; color:#F8FAFC; line-height:1.5; font-style:italic; padding:12px 14px; background:rgba(0,0,0,0.25); border-radius:10px; border-left:3px solid #8B5CF6;">
                        "{clean_single_review}"
                    </div>
                    <div style="display:flex; gap:10px; margin-top:16px; flex-wrap:wrap;">
                        <div>
                            <span style="font-size:0.75rem; color:#64748B; margin-right:4px;">Sentiment:</span>
                            <span class="badge {sent_badge_class}">{result['Sentiment']}</span>
                        </div>
                        <div>
                            <span style="font-size:0.75rem; color:#64748B; margin-right:4px;">Priority:</span>
                            <span class="badge {prio_badge_class}">{result['Priority']}</span>
                        </div>
                        <div>
                            <span style="font-size:0.75rem; color:#64748B; margin-right:4px;">Confidence:</span>
                            <span style="color:#C4B5FD; font-weight:700; font-size:0.85rem;">{result['Confidence']:.1%}</span>
                        </div>
                    </div>
                    <div style="margin-top:12px; font-size:0.85rem; color:#94A3B8;">
                        <strong>Detected Topics:</strong> <span style="color:#E2E8F0;">{result['Topic']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_rev2:
                st.markdown(f"""
                <div class="revio-card">
                    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:#38BDF8; font-weight:700; margin-bottom:8px;">Generated Resolution Response</div>
                    <div style="font-size:0.95rem; color:#E2E8F0; line-height:1.6; padding:14px; background:rgba(30, 41, 59, 0.5); border-radius:10px; border:1px solid rgba(56, 189, 248, 0.2);">
                        {result['Suggested Response']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.warning("Please enter review text to test the model.")

# Dataset analysis or Empty State display
if df.empty:
    if view_mode == "⚡ Live Customer Reviews":
        st.markdown(f"""
        <div style="margin: 40px auto; max-width: 720px; padding: 40px 30px; text-align: center; background: linear-gradient(145deg, rgba(20, 27, 45, 0.7) 0%, rgba(13, 18, 30, 0.7) 100%); border: 1px dashed rgba(139, 92, 246, 0.35); border-radius: 20px;">
            <div style="font-size: 3rem; margin-bottom: 12px;">📬</div>
            <h3 style="color: #F8FAFC; font-weight: 700; margin-bottom: 8px;">No Live Customer Reviews Yet</h3>
            <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.6; max-width: 520px; margin: 0 auto 20px auto;">
                Your customer review database for <strong>{auth_business_name}</strong> is currently empty.
                Once customers submit feedback through your public review widget, their entries will automatically appear here with AI sentiment analysis and suggested responses.
            </p>
            <div style="display:inline-flex; gap:12px; align-items:center;">
                <a href="?page=submit_review&biz={auth_business_id}" target="_blank" style="display:inline-block; padding: 10px 22px; background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 100%); color: white; text-decoration: none; border-radius: 10px; font-weight: 600; font-size: 0.88rem; box-shadow: 0 4px 14px rgba(124, 58, 237, 0.4);">
                    🔗 Open Customer Review Form
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif view_mode == "📥 CSV Data Import":
        st.info("Upload a CSV file with a `review` column above to view analytics.")
    elif view_mode == "🧪 Demonstration Dataset":
        st.info("Check 'Load Demonstration Reviews' in the sidebar to view sample data analytics.")
else:
    if "review" not in df.columns:
        st.error("Dataset must contain a 'review' column.")
    else:
        df = df.dropna(subset=["review"]).copy()
        df["review"] = df["review"].astype(str)

        # For live reviews, preserve existing response_status, approved_response, and suggested_response
        has_approval_workflow = "response_status" in df.columns

        results = []
        for _, row in df.iterrows():
            rev_txt = row["review"]
            # If review already has AI fields pre-computed from database, preserve them
            if has_approval_workflow and pd.notna(row.get("sentiment")):
                results.append({
                    "Sentiment": row.get("sentiment"),
                    "Confidence": row.get("confidence", 0.0),
                    "Topic": row.get("topic", "General Feedback"),
                    "Priority": row.get("priority", "Low"),
                    "Suggested Response": row.get("suggested_response", "")
                })
            else:
                results.append(analyze_review(rev_txt))

        results_df = pd.DataFrame(results)
        final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        # Ensure response_status column exists in final_df for filtering
        if "response_status" not in final_df.columns:
            final_df["response_status"] = "pending_approval"
        if "approved_response" not in final_df.columns:
            final_df["approved_response"] = None

        # Urgent Complaints Section
        urgent_reviews = final_df[final_df["Priority"] == "High"]
        auth_user_id = st.session_state.get("user_id")

        st.markdown("""
        <div class="section-header">
            <span class="section-icon">🚨</span>
            <h3 class="section-title">Critical & Urgent Escalations</h3>
        </div>
        """, unsafe_allow_html=True)

        if not urgent_reviews.empty:
            st.markdown(f"""
            <div class="urgent-banner">
                <span style="font-size:1.3rem;">⚠️</span>
                <div>
                    <strong>Attention Needed:</strong> {len(urgent_reviews)} high-priority complaints have been detected and require immediate intervention.
                </div>
            </div>
            """, unsafe_allow_html=True)

            for index, row in urgent_reviews.iterrows():
                review_id_label = row.get('review_id', index + 1)
                raw_id = row.get('id')
                status = row.get('response_status', 'pending_approval')
                is_live_review = (view_mode == "⚡ Live Customer Reviews") and (raw_id is not None)

                urg_status_badge = (
                    "<span style='background:rgba(16, 185, 129, 0.2); color:#34D399; border:1px solid rgba(16, 185, 129, 0.4); padding:2px 8px; border-radius:9999px; font-size:0.72rem; font-weight:700;'>✓ APPROVED</span>"
                    if status == "approved"
                    else "<span style='background:rgba(245, 158, 11, 0.2); color:#FBBF24; border:1px solid rgba(245, 158, 11, 0.4); padding:2px 8px; border-radius:9999px; font-size:0.72rem; font-weight:700;'>⏳ PENDING APPROVAL</span>"
                )

                urg_title = f"⚠️ Review #{review_id_label} — Topic: {row['Topic']} | Status: {'APPROVED' if status == 'approved' else 'PENDING'}"

                with st.expander(urg_title):
                    u_col1, u_col2 = st.columns([1, 1.2])

                    with u_col1:
                        st.markdown(f"**Customer Review Feedback:**\n> {row['review']}")
                        st.markdown(f"""
                        <div style="margin-top:10px;">
                            <span style="font-size:0.8rem; color:#94A3B8;">Sentiment:</span> <span class='badge badge-neg'>{row['Sentiment']}</span> &nbsp;|&nbsp; 
                            <span style="font-size:0.8rem; color:#94A3B8;">Priority:</span> <span class='badge badge-high'>{row['Priority']}</span> &nbsp;|&nbsp; 
                            <span style="font-size:0.8rem; color:#94A3B8;">Status:</span> {urg_status_badge}
                        </div>
                        """, unsafe_allow_html=True)

                        if status == "approved":
                            approved_by_label = row.get("approved_by_username") or f"Employee #{row.get('approved_by', 'ID')}"
                            approved_at_label = row.get("approved_at") or "Recently"
                            st.markdown(f"""
                            <div style="margin-top:12px; padding:10px 14px; background:rgba(16, 185, 129, 0.08); border:1px solid rgba(16, 185, 129, 0.25); border-radius:8px;">
                                <div style="font-size:0.75rem; color:#34D399; font-weight:700; text-transform:uppercase;">Internal Approval Record</div>
                                <div style="font-size:0.82rem; color:#E2E8F0; margin-top:2px;">Approved by <strong>{approved_by_label}</strong> on {approved_at_label}</div>
                                <div style="font-size:0.72rem; color:#94A3B8; margin-top:2px;">(Internal employee resolution sign-off; not dispatched externally)</div>
                            </div>
                            """, unsafe_allow_html=True)

                    with u_col2:
                        current_suggested = row.get("suggested_response", "")
                        approved_resp_text = row.get("approved_response")

                        if status == "approved":
                            st.markdown("**Final Approved Action & Response:**")
                            st.markdown(f"""
                            <div style="background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.35); border-radius:10px; padding:14px; font-size:0.92rem; color:#E2E8F0; line-height:1.5;">
                                {approved_resp_text or current_suggested}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("**AI-Suggested Action & Response:**")
                            st.markdown(f"""
                            <div style="background:rgba(30, 41, 59, 0.4); border:1px solid rgba(139, 92, 246, 0.25); border-radius:10px; padding:12px; font-size:0.92rem; color:#E2E8F0; line-height:1.5; margin-bottom:12px;">
                                {current_suggested}
                            </div>
                            """, unsafe_allow_html=True)

                            if is_live_review:
                                st.markdown("<p style='font-size:0.78rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:6px;'>Employee Approval Actions</p>", unsafe_allow_html=True)
                                
                                edit_expander = st.checkbox(f"✏️ Edit response before approving", key=f"urg_edit_chk_{raw_id}")
                                response_to_approve = current_suggested

                                if edit_expander:
                                    response_to_approve = st.text_area(
                                        "Draft Response",
                                        value=current_suggested,
                                        height=100,
                                        key=f"urg_edit_area_{raw_id}",
                                        label_visibility="collapsed"
                                    )

                                act_col1, act_col2 = st.columns([1, 1.2])

                                with act_col1:
                                    if st.button("🔄 Regenerate Suggestion", key=f"urg_regen_btn_{raw_id}", use_container_width=True):
                                        try:
                                            topics_list = [t.strip() for t in str(row['Topic']).split(",")]
                                            new_gen_response = generate_response(
                                                row['review'],
                                                row['Sentiment'],
                                                topics_list,
                                                row['Priority']
                                            )
                                            update_suggested_response(
                                                review_id=int(raw_id),
                                                business_id=auth_business_id,
                                                new_suggested_response=new_gen_response
                                            )
                                            st.success("New suggestion generated!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error regenerating: {str(e)}")

                                with act_col2:
                                    confirm_approval = st.checkbox("Confirm approval", key=f"urg_confirm_{raw_id}")
                                    if st.button("✅ Approve Response", key=f"urg_appr_btn_{raw_id}", disabled=not confirm_approval, use_container_width=True):
                                        if not response_to_approve or not response_to_approve.strip():
                                            st.error("Cannot approve an empty response.")
                                        elif not auth_user_id:
                                            st.error("Authentication required to approve responses.")
                                        else:
                                            try:
                                                approve_review_response(
                                                    review_id=int(raw_id),
                                                    business_id=auth_business_id,
                                                    employee_user_id=int(auth_user_id),
                                                    approved_response=response_to_approve.strip()
                                                )
                                                st.success("High-priority response approved!")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Approval failed: {str(e)}")
                            else:
                                st.caption("ℹ️ Response approval workflow is active in Live Customer Reviews mode.")

        # Business Intelligence Dashboard Metrics
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">📊</span>
            <h3 class="section-title">Executive Performance Overview</h3>
        </div>
        """, unsafe_allow_html=True)

        total_cnt = len(final_df)
        pos_cnt = (final_df["Sentiment"] == "positive").sum()
        neu_cnt = (final_df["Sentiment"] == "neutral").sum()
        neg_cnt = (final_df["Sentiment"] == "negative").sum()
        prio_cnt = (final_df["Priority"] == "High").sum()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
            <div class="metric-card total">
                <div class="metric-label">
                    <span>Total Reviews</span>
                    <span style="color:#818CF8;">📁</span>
                </div>
                <div class="metric-value">{total_cnt}</div>
                <div class="metric-sub">Processed Ingestion</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card positive">
                <div class="metric-label">
                    <span>Positive</span>
                    <span style="color:#34D399;">👍</span>
                </div>
                <div class="metric-value">{pos_cnt}</div>
                <div class="metric-sub">{pos_cnt/total_cnt:.1%} of overall volume</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card neutral">
                <div class="metric-label">
                    <span>Neutral</span>
                    <span style="color:#FBBF24;">⚖️</span>
                </div>
                <div class="metric-value">{neu_cnt}</div>
                <div class="metric-sub">{neu_cnt/total_cnt:.1%} of overall volume</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card negative">
                <div class="metric-label">
                    <span>Negative</span>
                    <span style="color:#F87171;">👎</span>
                </div>
                <div class="metric-value">{neg_cnt}</div>
                <div class="metric-sub">{neg_cnt/total_cnt:.1%} of overall volume</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="metric-card priority">
                <div class="metric-label">
                    <span>High Priority</span>
                    <span style="color:#F472B6;">🚨</span>
                </div>
                <div class="metric-value">{prio_cnt}</div>
                <div class="metric-sub">Action required</div>
            </div>
            """, unsafe_allow_html=True)

        # Model Performance & Confusion Matrix
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">🧠</span>
            <h3 class="section-title">Sentiment Model Performance & Evaluation</h3>
        </div>
        """, unsafe_allow_html=True)

        eval_col_left, eval_col_right = st.columns([1, 1])

        with eval_col_left:
            st.caption("Standardized benchmark testing on 30 ground-truth validation reviews:")
            
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown(f"""
                <div class="revio-card" style="margin-bottom:14px; text-align:center;">
                    <div style="font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:700;">Accuracy</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#38BDF8; margin:6px 0;">{model_accuracy:.1%}</div>
                    <div style="font-size:0.75rem; color:#64748B;">Correct Predictions</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="revio-card" style="text-align:center;">
                    <div style="font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:700;">Recall</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#818CF8; margin:6px 0;">{model_recall:.1%}</div>
                    <div style="font-size:0.75rem; color:#64748B;">True Positive Rate</div>
                </div>
                """, unsafe_allow_html=True)

            with pcol2:
                st.markdown(f"""
                <div class="revio-card" style="margin-bottom:14px; text-align:center;">
                    <div style="font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:700;">Precision</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#A78BFA; margin:6px 0;">{model_precision:.1%}</div>
                    <div style="font-size:0.75rem; color:#64748B;">Positive Predictive Value</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="revio-card" style="text-align:center;">
                    <div style="font-size:0.75rem; text-transform:uppercase; color:#94A3B8; font-weight:700;">F1-Score</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#34D399; margin:6px 0;">{model_f1:.1%}</div>
                    <div style="font-size:0.75rem; color:#64748B;">Harmonic Mean</div>
                </div>
                """, unsafe_allow_html=True)

        with eval_col_right:
            labels_cm = ["Positive", "Neutral", "Negative"]
            cm_fig = px.imshow(
                model_cm,
                x=labels_cm,
                y=labels_cm,
                text_auto=True,
                color_continuous_scale=[
                    [0.0, "#0F172A"],
                    [0.5, "#4338CA"],
                    [1.0, "#8B5CF6"]
                ],
                labels={
                    "x": "Predicted Sentiment",
                    "y": "Actual Sentiment",
                    "color": "Reviews"
                },
                title="Confusion Matrix (Ground Truth vs Model Prediction)"
            )
            cm_fig = apply_revio_plotly_theme(cm_fig, height=315)
            cm_fig.update_traces(
                textfont=dict(size=14, color="#FFFFFF", family="Plus Jakarta Sans"),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            )
            st.plotly_chart(cm_fig, use_container_width=True)

        # Topic & Distribution Analytics
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">📈</span>
            <h3 class="section-title">Sentiment & Issue Breakdown</h3>
        </div>
        """, unsafe_allow_html=True)

        chart_row1, chart_row2, chart_row3 = st.columns([1.2, 1, 1])

        with chart_row1:
            topic_data = (
                final_df["Topic"]
                .str.split(", ")
                .explode()
                .value_counts()
                .reset_index()
            )
            topic_data.columns = ["Topic", "Count"]

            topic_fig = px.bar(
                topic_data,
                x="Topic",
                y="Count",
                title="Customer Issues by Category",
                text="Count",
                color="Count",
                color_continuous_scale=[[0.0, "#3B82F6"], [1.0, "#8B5CF6"]]
            )
            topic_fig = apply_revio_plotly_theme(topic_fig, height=340)
            topic_fig.update_traces(
                textposition='outside',
                textfont=dict(color="#CBD5E1", size=11, family="Plus Jakarta Sans"),
                marker_line_color='rgba(255,255,255,0.1)',
                marker_line_width=1,
                hovertemplate="<b>%{x}</b><br>Incidents: %{y}<extra></extra>"
            )
            topic_fig.update_coloraxes(showscale=False)
            st.plotly_chart(topic_fig, use_container_width=True)

        with chart_row2:
            sentiment_counts = (
                final_df["Sentiment"].value_counts().reset_index()
            )
            sentiment_counts.columns = ["Sentiment", "Count"]

            color_map = {
                "positive": "#10B981",
                "neutral": "#F59E0B",
                "negative": "#EF4444"
            }

            fig1 = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                title="Sentiment Distribution",
                hole=0.55,
                color="Sentiment",
                color_discrete_map=color_map
            )
            fig1 = apply_revio_plotly_theme(fig1, height=340)
            fig1.update_traces(
                textinfo='percent+label',
                textfont=dict(color="#FFFFFF", size=12, family="Plus Jakarta Sans"),
                marker=dict(line=dict(color='#0F172A', width=2)),
                hovertemplate="<b>%{label}</b><br>Reviews: %{value} (%{percent})<extra></extra>"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with chart_row3:
            priority_counts = (
                final_df["Priority"].value_counts().reset_index()
            )
            priority_counts.columns = ["Priority", "Count"]

            priority_color_map = {
                "High": "#EF4444",
                "Medium": "#F59E0B",
                "Low": "#3B82F6"
            }

            fig2 = px.bar(
                priority_counts,
                x="Priority",
                y="Count",
                title="Priority Severity Spread",
                text="Count",
                color="Priority",
                color_discrete_map=priority_color_map
            )
            fig2 = apply_revio_plotly_theme(fig2, height=340)
            fig2.update_traces(
                textposition='outside',
                textfont=dict(color="#CBD5E1", size=11, family="Plus Jakarta Sans"),
                marker_line_color='rgba(255,255,255,0.1)',
                marker_line_width=1,
                hovertemplate="<b>%{x} Priority</b><br>Count: %{y}<extra></extra>"
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Interactive Filtering & Table Exploration
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">🔍</span>
            <h3 class="section-title">Review Explorer & Triage Filters</h3>
        </div>
        """, unsafe_allow_html=True)

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                ["All", "positive", "neutral", "negative"]
            )

        with filter_col2:
            priority_filter = st.selectbox(
                "Filter by Priority",
                ["All", "High", "Medium", "Low"]
            )

        with filter_col3:
            topic_options = ["All"] + sorted(
                final_df["Topic"]
                .str.split(", ")
                .explode()
                .dropna()
                .unique()
                .tolist()
            )
            topic_filter = st.selectbox(
                "Filter by Topic",
                topic_options
            )

        with filter_col4:
            status_filter = st.selectbox(
                "Filter by Approval Status",
                ["All", "Pending Approval", "Approved"]
            )

        filtered_df = final_df.copy()

        if sentiment_filter != "All":
            filtered_df = filtered_df[
                filtered_df["Sentiment"] == sentiment_filter
            ]

        if priority_filter != "All":
            filtered_df = filtered_df[
                filtered_df["Priority"] == priority_filter
            ]

        if topic_filter != "All":
            filtered_df = filtered_df[
                filtered_df["Topic"].str.split(", ").apply(
                    lambda topics: topic_filter in topics
                )
            ]

        if status_filter == "Pending Approval":
            filtered_df = filtered_df[
                filtered_df["response_status"] == "pending_approval"
            ]
        elif status_filter == "Approved":
            filtered_df = filtered_df[
                filtered_df["response_status"] == "approved"
            ]

        st.markdown(f"""
        <div style="margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.9rem; color:#94A3B8;">
                Displaying <b style="color:#818CF8;">{len(filtered_df)}</b> of {len(final_df)} filtered customer feedback items
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

        # Response Generator Accordion & Workflow
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">💬</span>
            <h3 class="section-title">Customer Resolution & Response Approval Workflow</h3>
        </div>
        """, unsafe_allow_html=True)

        auth_user_id = st.session_state.get("user_id")

        for index, row in filtered_df.iterrows():
            prio = row['Priority']
            icon = "🔴" if prio == "High" else ("🟡" if prio == "Medium" else "🔵")
            rev_id = row.get('review_id', index + 1)
            raw_id = row.get('id')  # Real integer PK from SQLite if live reviews
            status = row.get('response_status', 'pending_approval')
            is_live_review = (view_mode == "⚡ Live Customer Reviews") and (raw_id is not None)

            status_badge_html = (
                "<span style='background:rgba(16, 185, 129, 0.2); color:#34D399; border:1px solid rgba(16, 185, 129, 0.4); padding:2px 8px; border-radius:9999px; font-size:0.72rem; font-weight:700;'>✓ APPROVED</span>"
                if status == "approved"
                else "<span style='background:rgba(245, 158, 11, 0.2); color:#FBBF24; border:1px solid rgba(245, 158, 11, 0.4); padding:2px 8px; border-radius:9999px; font-size:0.72rem; font-weight:700;'>⏳ PENDING APPROVAL</span>"
            )

            expander_title = f"{icon} Review #{rev_id} — {prio} Priority | Topic: {row['Topic']} | Status: {'APPROVED' if status == 'approved' else 'PENDING'}"

            with st.expander(expander_title):
                r_col1, r_col2 = st.columns([1, 1.2])

                with r_col1:
                    st.markdown("**Customer Review Feedback:**")
                    st.markdown(f"> *\"{row['review']}\"*")
                    st.markdown(f"""
                    <div style="margin-top:10px;">
                        <span style="font-size:0.8rem; color:#94A3B8;">Sentiment:</span> 
                        <b style="color:#CBD5E1;">{row['Sentiment']}</b> &nbsp;|&nbsp; 
                        <span style="font-size:0.8rem; color:#94A3B8;">Priority:</span> 
                        <b style="color:#CBD5E1;">{row['Priority']}</b> &nbsp;|&nbsp; 
                        <span style="font-size:0.8rem; color:#94A3B8;">Status:</span> {status_badge_html}
                    </div>
                    """, unsafe_allow_html=True)

                    if status == "approved":
                        approved_by_label = row.get("approved_by_username") or f"Employee #{row.get('approved_by', 'ID')}"
                        approved_at_label = row.get("approved_at") or "Recently"
                        st.markdown(f"""
                        <div style="margin-top:12px; padding:10px 14px; background:rgba(16, 185, 129, 0.08); border:1px solid rgba(16, 185, 129, 0.25); border-radius:8px;">
                            <div style="font-size:0.75rem; color:#34D399; font-weight:700; text-transform:uppercase;">Internal Approval Record</div>
                            <div style="font-size:0.82rem; color:#E2E8F0; margin-top:2px;">Approved by <strong>{approved_by_label}</strong> on {approved_at_label}</div>
                            <div style="font-size:0.72rem; color:#94A3B8; margin-top:2px;">(Internal employee resolution sign-off; not dispatched externally)</div>
                        </div>
                        """, unsafe_allow_html=True)

                with r_col2:
                    current_suggested = row.get("suggested_response", "")
                    approved_resp_text = row.get("approved_response")

                    if status == "approved":
                        st.markdown("**Final Approved Response:**")
                        st.markdown(f"""
                        <div style="background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.35); border-radius:10px; padding:14px; font-size:0.92rem; color:#E2E8F0; line-height:1.5;">
                            {approved_resp_text or current_suggested}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("**AI-Generated Suggested Response:**")
                        st.markdown(f"""
                        <div style="background:rgba(30, 41, 59, 0.4); border:1px solid rgba(139, 92, 246, 0.25); border-radius:10px; padding:12px; font-size:0.92rem; color:#E2E8F0; line-height:1.5; margin-bottom:12px;">
                            {current_suggested}
                        </div>
                        """, unsafe_allow_html=True)

                        if is_live_review:
                            # Actions: Edit, Regenerate, Approve
                            st.markdown("<p style='font-size:0.78rem; font-weight:700; color:#94A3B8; text-transform:uppercase; margin-bottom:6px;'>Employee Approval Actions</p>", unsafe_allow_html=True)
                            
                            edit_expander = st.checkbox(f"✏️ Edit response before approving", key=f"edit_chk_{raw_id}")
                            response_to_approve = current_suggested

                            if edit_expander:
                                response_to_approve = st.text_area(
                                    "Draft Response",
                                    value=current_suggested,
                                    height=100,
                                    key=f"edit_area_{raw_id}",
                                    label_visibility="collapsed"
                                )

                            act_col1, act_col2 = st.columns([1, 1.2])

                            with act_col1:
                                if st.button("🔄 Regenerate Suggestion", key=f"regen_btn_{raw_id}", use_container_width=True):
                                    try:
                                        # Use existing response_generator function
                                        topics_list = [t.strip() for t in str(row['Topic']).split(",")]
                                        new_gen_response = generate_response(
                                            row['review'],
                                            row['Sentiment'],
                                            topics_list,
                                            row['Priority']
                                        )
                                        update_suggested_response(
                                            review_id=int(raw_id),
                                            business_id=auth_business_id,
                                            new_suggested_response=new_gen_response
                                        )
                                        st.success("New suggestion generated!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error regenerating: {str(e)}")

                            with act_col2:
                                confirm_approval = st.checkbox("Confirm approval", key=f"confirm_{raw_id}")
                                if st.button("✅ Approve Response", key=f"appr_btn_{raw_id}", disabled=not confirm_approval, use_container_width=True):
                                    if not response_to_approve or not response_to_approve.strip():
                                        st.error("Cannot approve an empty response.")
                                    elif not auth_user_id:
                                        st.error("Authentication required to approve responses.")
                                    else:
                                        try:
                                            approve_review_response(
                                                review_id=int(raw_id),
                                                business_id=auth_business_id,
                                                employee_user_id=int(auth_user_id),
                                                approved_response=response_to_approve.strip()
                                            )
                                            st.success("Response approved and logged!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Approval failed: {str(e)}")
                        else:
                            st.caption("ℹ️ Response approval workflow is active in Live Customer Reviews mode.")

        # Download Section
        st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
        csv_data = final_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Analysis Report (CSV)",
            data=csv_data,
            file_name="revio_ai_analysis.csv",
            mime="text/csv"
        )
