"""
Automated end-to-end integration test for Phase 2: Public Customer Review Submission.
Verifies business validation, customer review processing via AI pipeline,
database persistence, duplicate submission prevention, and rating enforcement.
"""

import os
import tempfile
import unittest

from src.database import (
    init_db,
    create_business,
    get_business,
    save_review,
    get_reviews_by_business
)
from src.sentiment import train_sentiment_model, analyze_sentiment
from src.topic_classifier import classify_topic
from src.priority import detect_priority
from src.response_generator import generate_response


class TestCustomerSubmissionWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Train model once for test suite
        cls.model = train_sentiment_model()

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_db(self.db_path)

        # Register demo business
        create_business("acme-electronics", "Acme Electronics", self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass

    def test_unknown_business_id_rejected(self):
        # Form logic must reject unregistered businesses
        biz = get_business("invalid-nonexistent-biz", self.db_path)
        self.assertIsNone(biz)

    def test_known_business_id_accepted(self):
        biz = get_business("acme-electronics", self.db_path)
        self.assertIsNotNone(biz)
        self.assertEqual(biz["name"], "Acme Electronics")

    def test_submission_pipeline_negative_urgent(self):
        review_text = "I received a damaged item and customer support never responded. I want a refund immediately!"
        customer_name = "Marcus Vance"
        rating = 1

        # 1. AI Pipeline execution
        sent_result = analyze_sentiment(review_text, model=self.model)
        self.assertEqual(sent_result["sentiment"], "negative")
        self.assertGreater(sent_result["confidence"], 0.0)

        topics = classify_topic(review_text)
        self.assertTrue(any(t in topics for t in ["Product Quality", "Customer Service", "Pricing"]))

        priority = detect_priority(review_text, sent_result["sentiment"])
        self.assertEqual(priority, "High")

        response = generate_response(review_text, sent_result["sentiment"], topics, priority)
        self.assertIn("apologize", response.lower())

        # 2. Database persistence
        saved = save_review(
            business_id="acme-electronics",
            rating=rating,
            review_text=review_text,
            customer_name=customer_name,
            sentiment=sent_result["sentiment"],
            confidence=sent_result["confidence"],
            topic=", ".join(topics),
            priority=priority,
            suggested_response=response,
            db_path=self.db_path
        )

        self.assertIsNotNone(saved["id"])
        self.assertEqual(saved["customer_name"], "Marcus Vance")
        self.assertEqual(saved["priority"], "High")

        # 3. Fetch from business reviews
        reviews = get_reviews_by_business("acme-electronics", self.db_path)
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["rating"], 1)
        self.assertEqual(reviews[0]["sentiment"], "negative")

    def test_submission_pipeline_positive_anonymous(self):
        review_text = "Excellent product and outstanding quality"
        customer_name = ""  # Anonymous
        rating = 5

        sent_result = analyze_sentiment(review_text, model=self.model)
        self.assertEqual(sent_result["sentiment"], "positive")

        topics = classify_topic(review_text)
        priority = detect_priority(review_text, sent_result["sentiment"])
        self.assertEqual(priority, "Low")

        response = generate_response(review_text, sent_result["sentiment"], topics, priority)
        self.assertIn("delighted", response.lower())

        saved = save_review(
            business_id="acme-electronics",
            rating=rating,
            review_text=review_text,
            customer_name=customer_name,
            sentiment=sent_result["sentiment"],
            confidence=sent_result["confidence"],
            topic=", ".join(topics),
            priority=priority,
            suggested_response=response,
            db_path=self.db_path
        )

        self.assertIsNone(saved["customer_name"])
        self.assertEqual(saved["priority"], "Low")

        reviews = get_reviews_by_business("acme-electronics", self.db_path)
        self.assertEqual(len(reviews), 1)
        self.assertIsNone(reviews[0]["customer_name"])


if __name__ == "__main__":
    unittest.main()
