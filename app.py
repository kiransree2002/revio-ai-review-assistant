
import streamlit as st
import pandas as pd
import plotly.express as px

from src.sentiment import train_sentiment_model, analyze_sentiment
from src.topic_classifier import classify_topic
from src.priority import detect_priority
from src.response_generator import generate_response
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
# Page configuration
st.set_page_config(
    page_title="Revio AI | Review Intelligence",
    page_icon="⭐",
    layout="wide"
)

# Title
st.title("⭐ Revio AI")
st.subheader("Intelligent Review Analysis & Customer Resolution Assistant")

st.write(
    "Analyze customer feedback, understand sentiment, "
    "identify issues, prioritize complaints and generate "
    "suggested responses."
)

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


# Sidebar
st.sidebar.header("Review Input")

input_method = st.sidebar.radio(
    "Choose input method",
    ["Sample Dataset", "Upload CSV", "Enter Review"]
)


# Load reviews
if input_method == "Sample Dataset":

    df = pd.read_csv("data/sample_reviews.csv")

elif input_method == "Upload CSV":

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.DataFrame()

else:

    review_text = st.sidebar.text_area(
        "Enter customer review"
    )

    if st.sidebar.button("Analyze Review"):
        if review_text.strip():
            result = analyze_review(review_text)

            st.subheader("Review Analysis")
            st.write("**Customer Review:**", review_text)

            for key, value in result.items():
                st.write(f"**{key}:** {value}")

        else:
            st.warning("Please enter a review.")

    df = pd.DataFrame()


# Dataset analysis
if not df.empty:

    if "review" not in df.columns:
        st.error("CSV must contain a 'review' column.")
    else:

        df = df.dropna(subset=["review"]).copy()
        df["review"] = df["review"].astype(str)

        results = df["review"].apply(analyze_review)

        results_df = pd.DataFrame(results.tolist())

        final_df = pd.concat(
            [df.reset_index(drop=True), results_df],
            axis=1
        )

        
        # Urgent Complaints
        st.subheader("🚨 Urgent Customer Complaints")

        urgent_reviews = final_df[
            final_df["Priority"] == "High"
        ]

        if not urgent_reviews.empty:
            st.warning(
                f"{len(urgent_reviews)} high-priority complaints "
                "require attention."
            )

            for _, row in urgent_reviews.iterrows():
                with st.expander(
                    f"Review ID: {row.get('review_id', 'N/A')} "
                    f"— {row['Topic']}"
                ):
                    st.write("**Customer Review:**", row["review"])
                    st.write("**Sentiment:**", row["Sentiment"])
                    st.write("**Suggested Action:**", row["Suggested Response"])
        else:
            st.success("No high-priority complaints found.")

        # Metrics
        st.divider()
        st.header("Business Intelligence Dashboard")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Total Reviews", len(final_df))

        col2.metric(
            "Positive Reviews",
            (final_df["Sentiment"] == "positive").sum()
        )

        col3.metric(
            "Neutral Reviews",
            (final_df["Sentiment"] == "neutral").sum()
        )

        col4.metric(
            "Negative Reviews",
            (final_df["Sentiment"] == "negative").sum()
        )

        col5.metric(
            "High Priority Issues",
            (final_df["Priority"] == "High").sum()
        )
                # Model Performance
        st.subheader("Sentiment Model Performance")

        st.caption(
            "Evaluation results based on 30 manually labelled test reviews."
        )

        perf1, perf2, perf3, perf4 = st.columns(4)

        perf1.metric("Accuracy", f"{model_accuracy:.1%}")
        perf2.metric("Precision", f"{model_precision:.1%}")
        perf3.metric("Recall", f"{model_recall:.1%}")
        perf4.metric("F1-Score", f"{model_f1:.1%}")
                # Confusion Matrix
        st.subheader("Sentiment Confusion Matrix")

        cm_fig = px.imshow(
            model_cm,
            x=["Positive", "Neutral", "Negative"],
            y=["Positive", "Neutral", "Negative"],
            text_auto=True,
            color_continuous_scale="Blues",
            labels={
                "x": "Predicted Sentiment",
                "y": "Actual Sentiment",
                "color": "Reviews"
            }
        )

        st.plotly_chart(cm_fig, use_container_width=True)
        # Charts

        # Topic Analytics
        st.subheader("Customer Issue Categories")

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
            title="Customer Issues by Topic",
            text="Count"
        )

        st.plotly_chart(
            topic_fig,
            use_container_width=True
        )

        st.subheader("Review Analytics")

        chart1, chart2 = st.columns(2)

        with chart1:
            sentiment_counts = (
                final_df["Sentiment"].value_counts().reset_index()
            )
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig1 = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                title="Sentiment Distribution"
            )

            st.plotly_chart(fig1, use_container_width=True)

        with chart2:
            priority_counts = (
                final_df["Priority"].value_counts().reset_index()
            )
            priority_counts.columns = ["Priority", "Count"]

            fig2 = px.bar(
                priority_counts,
                x="Priority",
                y="Count",
                title="Complaint Priority Distribution"
            )

            st.plotly_chart(fig2, use_container_width=True)

        # Review table

        # Review Filters
        st.subheader("Filter Customer Reviews")

        filter_col1, filter_col2, filter_col3 = st.columns(3)

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

        st.write(f"Showing {len(filtered_df)} matching reviews.")

        st.subheader("Analyzed Customer Reviews")

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

        # Response generator
        st.subheader("Suggested Customer Responses")

        for index, row in filtered_df.iterrows():
            with st.expander(
                f"Review {index + 1} — {row['Priority']} Priority"
            ):

                st.write("**Customer Review:**", row["review"])

                st.write("**Sentiment:**", row["Sentiment"])

                st.write("**Topic:**", row["Topic"])

                st.write(
                    "**Suggested Response:**",
                    row["Suggested Response"]
                )

        # Download results
        csv_data = final_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Analysis Report",
            data=csv_data,
            file_name="revio_ai_analysis.csv",
            mime="text/csv"
        )
