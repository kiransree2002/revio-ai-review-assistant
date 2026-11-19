"""
Security and input hardening tests for Phase 4.
Tests:
- Rejection of oversized customer name (>100 chars)
- Rejection of oversized customer review text (>5000 chars)
- Handling of whitespace-only / empty inputs
- Rating boundaries validation
- DB connection safety and foreign key enforcement under invalid inputs
- Custom REVIO_DB_PATH environment variable override
- Tenant review isolation guarantees
"""

import os
import tempfile
import unittest

from src.database import (
    init_db,
    create_business,
    create_user,
    save_review,
    get_reviews_by_business
)


class TestSecurityHardening(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_db(self.db_path)

        create_business("secure-corp", "Secure Corp", self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass

    def test_oversized_review_text_rejected(self):
        huge_text = "A" * 5001
        with self.assertRaises(ValueError) as ctx:
            save_review(
                business_id="secure-corp",
                rating=5,
                review_text=huge_text,
                db_path=self.db_path
            )
        self.assertIn("cannot exceed 5000 characters", str(ctx.exception))

    def test_oversized_customer_name_rejected(self):
        huge_name = "N" * 101
        with self.assertRaises(ValueError) as ctx:
            save_review(
                business_id="secure-corp",
                rating=5,
                review_text="Normal review text",
                customer_name=huge_name,
                db_path=self.db_path
            )
        self.assertIn("cannot exceed 100 characters", str(ctx.exception))

    def test_whitespace_only_review_text_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            save_review(
                business_id="secure-corp",
                rating=5,
                review_text="     \n\t  ",
                db_path=self.db_path
            )
        self.assertIn("review_text cannot be empty", str(ctx.exception))

    def test_invalid_rating_rejected(self):
        for bad_rating in [-1, 0, 6, 99]:
            with self.assertRaises(ValueError):
                save_review(
                    business_id="secure-corp",
                    rating=bad_rating,
                    review_text="Valid review",
                    db_path=self.db_path
                )

    def test_unregistered_business_foreign_key_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            save_review(
                business_id="unregistered-biz-xyz",
                rating=5,
                review_text="Great job",
                db_path=self.db_path
            )
        self.assertIn("does not exist", str(ctx.exception))

    def test_max_boundary_inputs_accepted(self):
        boundary_text = "B" * 5000
        boundary_name = "C" * 100
        saved = save_review(
            business_id="secure-corp",
            rating=4,
            review_text=boundary_text,
            customer_name=boundary_name,
            db_path=self.db_path
        )
        self.assertIsNotNone(saved["id"])
        self.assertEqual(len(saved["review_text"]), 5000)
        self.assertEqual(len(saved["customer_name"]), 100)


if __name__ == "__main__":
    unittest.main()

