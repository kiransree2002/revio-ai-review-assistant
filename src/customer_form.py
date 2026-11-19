"""
Public customer review submission form component.
Provides a clean, distraction-free, embed-ready interface for customers
to rate and review a business, automatically running sentiment, topic,
priority, and response AI pipelines and persisting to the database.
"""

import streamlit as st
from src.database import get_business, save_review
from src.sentiment import analyze_sentiment
from src.topic_classifier import classify_topic
from src.priority import detect_priority
from src.response_generator import generate_response


def render_customer_review_form(business_id: str, model=None) -> None:
    """
    Renders the public customer review submission page.
    Validates business_id, collects rating/name/review, processes AI inference,
    persists to SQLite, and prevents duplicate submissions.
    """
    if not business_id or not business_id.strip():
        st.error("No business identified. Please provide a valid business link (e.g. ?biz=demo-corp).")
        return

    clean_biz_id = business_id.strip().lower()
    business = get_business(clean_biz_id)

    if not business:
        st.markdown(f"""
        <div style="max-width: 650px; margin: 40px auto; padding: 28px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 16px; text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 10px;">🏢 ❓</div>
            <h3 style="color: #F87171; margin: 0 0 10px 0;">Business Not Found</h3>
            <p style="color: #CBD5E1; font-size: 0.95rem; margin: 0;">
                The business identifier <code>{clean_biz_id}</code> is not registered in our system.
                Please verify the review link provided by the business.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Check if a review was just successfully submitted in this session
    submitted_key = f"review_submitted_{clean_biz_id}"
    if st.session_state.get(submitted_key, False):
        st.markdown(f"""
        <div style="max-width: 650px; margin: 50px auto; padding: 36px 30px; background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(15, 23, 42, 0.8) 100%); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 20px; text-align: center; box-shadow: 0 20px 35px -10px rgba(0, 0, 0, 0.5);">
            <div style="font-size: 3rem; margin-bottom: 12px;">🎉</div>
            <h2 style="color: #34D399; font-weight: 800; margin: 0 0 8px 0; font-family: 'Plus Jakarta Sans', sans-serif;">Thank You for Your Feedback!</h2>
            <p style="color: #E2E8F0; font-size: 1.05rem; line-height: 1.5; margin: 0 0 20px 0;">
                Your review for <strong>{business['name']}</strong> has been received and logged successfully. We appreciate your valuable feedback.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_c1, col_c2, col_c3 = st.columns([1, 1.2, 1])
        with col_c2:
            if st.button("✍️ Submit Another Review", use_container_width=True):
                st.session_state[submitted_key] = False
                st.rerun()
        return

    # Render Public Form
    st.markdown(f"""
    <div style="max-width: 680px; margin: 20px auto 10px auto; text-align: center;">
        <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.3); color: #C4B5FD; padding: 4px 14px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 12px;">
            ⭐ Verified Customer Feedback
        </div>
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin: 0; letter-spacing: -0.02em;">
            Review {business['name']}
        </h1>
        <p style="color: #94A3B8; font-size: 0.98rem; margin: 8px auto 24px auto; max-width: 500px;">
            We appreciate your candid thoughts. Your review helps us continuously improve our service and support.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        # Outer form container styling
        st.markdown("""
        <style>
        .review-form-card {
            max-width: 680px;
            margin: 0 auto;
            background: linear-gradient(145deg, rgba(17, 24, 39, 0.85) 0%, rgba(15, 23, 42, 0.85) 100%);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 20px;
            padding: 30px 36px;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
        }
        </style>
        """, unsafe_allow_html=True)

        with st.form(key=f"customer_review_form_{clean_biz_id}", clear_on_submit=False):
            # 1. Star Rating
            st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #F1F5F9; margin-bottom: 6px;'>Overall Rating <span style='color:#EF4444;'>*</span></p>", unsafe_allow_html=True)
            rating_options = [
                "⭐⭐⭐⭐⭐ (5 - Exceptional)",
                "⭐⭐⭐⭐ (4 - Good)",
                "⭐⭐⭐ (3 - Average)",
                "⭐⭐ (2 - Below Expectations)",
                "⭐ (1 - Very Poor)"
            ]
            selected_rating_label = st.radio(
                "Star Rating",
                options=rating_options,
                index=0,
                label_visibility="collapsed"
            )
            # Map choice back to integer (1-5)
            rating_map = {
                "⭐⭐⭐⭐⭐ (5 - Exceptional)": 5,
                "⭐⭐⭐⭐ (4 - Good)": 4,
                "⭐⭐⭐ (3 - Average)": 3,
                "⭐⭐ (2 - Below Expectations)": 2,
                "⭐ (1 - Very Poor)": 1
            }
            rating = rating_map[selected_rating_label]

            # 2. Customer Name (Optional)
            st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #F1F5F9; margin: 18px 0 6px 0;'>Your Name <span style='color: #64748B; font-weight: 400; font-size: 0.8rem;'>(Optional, max 100 characters)</span></p>", unsafe_allow_html=True)
            customer_name = st.text_input(
                "Customer Name",
                placeholder="e.g. Jane Doe (leave blank to stay anonymous)",
                max_chars=100,
                label_visibility="collapsed"
            )

            # 3. Review Text (Required)
            st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #F1F5F9; margin: 18px 0 6px 0;'>Your Detailed Feedback <span style='color:#EF4444;'>*</span> <span style='color: #64748B; font-weight: 400; font-size: 0.8rem;'>(max 5000 characters)</span></p>", unsafe_allow_html=True)
            review_text = st.text_area(
                "Review Text",
                placeholder="Share your experience with the product, delivery, quality, or customer support...",
                max_chars=5000,
                height=140,
                label_visibility="collapsed"
            )

            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

            # 4. Submit Button
            submit_button = st.form_submit_button("🚀 Submit Review", use_container_width=True)

            if submit_button:
                clean_review_text = review_text.strip() if review_text else ""
                clean_name = customer_name.strip() if customer_name else None

                if not clean_review_text:
                    st.error("Please provide review feedback text before submitting.")
                elif len(clean_review_text) > 5000:
                    st.error("Review feedback cannot exceed 5000 characters.")
                elif clean_name and len(clean_name) > 100:
                    st.error("Customer name cannot exceed 100 characters.")
                else:
                    try:
                        # Execute AI Pipeline
                        sentiment_result = analyze_sentiment(clean_review_text, model=model)
                        sentiment = sentiment_result["sentiment"]
                        confidence = sentiment_result["confidence"]

                        topics = classify_topic(clean_review_text)
                        topic_str = ", ".join(topics)

                        priority = detect_priority(clean_review_text, sentiment)

                        response = generate_response(
                            clean_review_text,
                            sentiment,
                            topics,
                            priority
                        )

                        # Save to database
                        save_review(
                            business_id=clean_biz_id,
                            rating=rating,
                            review_text=clean_review_text,
                            customer_name=clean_name,
                            sentiment=sentiment,
                            confidence=confidence,
                            topic=topic_str,
                            priority=priority,
                            suggested_response=response
                        )

                        # Mark session as submitted to prevent duplicate re-submissions on page refresh
                        st.session_state[submitted_key] = True
                        st.rerun()
                    except Exception as err:
                        st.error(f"Failed to record review: {str(err)}")

    # Footer note
    st.markdown("""
    <div style="max-width: 680px; margin: 24px auto 0 auto; text-align: center; color: #64748B; font-size: 0.8rem;">
        Powered by <strong>Revio AI</strong> — Intelligent Customer Resolution & Sentiment Automation.
    </div>
    """, unsafe_allow_html=True)

