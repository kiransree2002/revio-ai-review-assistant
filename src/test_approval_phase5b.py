"""
Automated unit and workflow tests for Phase 5B:
Employee-controlled AI response approval workflow.

Verifies:
1. New customer reviews receive 'pending_approval' status.
2. Employee can edit a suggested response.
3. Employee can approve a response with confirmation.
4. Empty responses cannot be approved.
5. Approval records the correct employee ID and timestamp.
6. An employee cannot approve another business's review (tenant isolation).
7. Unauthenticated/invalid user cannot approve responses.
8. Approved responses and timestamps are preserved correctly.
9. Existing database records survive migration (safe ALTER TABLE).
10. Regeneration does not overwrite an already approved response.
"""

import os
import tempfile
import unittest
import sqlite3

from src.database import (
    init_db,
    create_business,
    create_user,
    save_review,
    get_reviews_by_business,
    update_suggested_response,
    approve_review_response
)


class TestResponseApprovalWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_db(self.db_path)

        # Create Business A and Employee A
        create_business("biz-alpha", "Alpha Stores", self.db_path)
        self.user_a = create_user("biz-alpha", "alice", "AlicePass123", role="support", db_path=self.db_path)

        # Create Business B and Employee B
        create_business("biz-beta", "Beta Shop", self.db_path)
        self.user_b = create_user("biz-beta", "bob", "BobPass123", role="support", db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass

    def test_new_customer_review_receives_pending_approval_status(self):
        saved = save_review(
            business_id="biz-alpha",
            rating=2,
            review_text="My order was delayed and arrived damaged.",
            sentiment="negative",
            priority="High",
            suggested_response="We apologize for the delayed and damaged delivery.",
            db_path=self.db_path
        )
        self.assertEqual(saved["response_status"], "pending_approval")
        self.assertIsNone(saved["approved_response"])
        self.assertIsNone(saved["approved_by"])

        # Fetch from database
        reviews = get_reviews_by_business("biz-alpha", self.db_path)
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["response_status"], "pending_approval")
        self.assertIsNone(reviews[0]["approved_response"])

    def test_employee_can_edit_suggested_response(self):
        saved = save_review(
            business_id="biz-alpha",
            rating=1,
            review_text="Nobody answered my complaint.",
            sentiment="negative",
            priority="High",
            suggested_response="Original AI generated response",
            db_path=self.db_path
        )

        # Employee edits response
        updated = update_suggested_response(
            review_id=saved["id"],
            business_id="biz-alpha",
            new_suggested_response="Edited response drafted by employee.",
            db_path=self.db_path
        )
        self.assertEqual(updated["suggested_response"], "Edited response drafted by employee.")

        # Status should still be pending_approval until explicitly approved
        reviews = get_reviews_by_business("biz-alpha", self.db_path)
        self.assertEqual(reviews[0]["response_status"], "pending_approval")
        self.assertEqual(reviews[0]["suggested_response"], "Edited response drafted by employee.")
        self.assertIsNone(reviews[0]["approved_response"])

    def test_employee_can_approve_response(self):
        saved = save_review(
            business_id="biz-alpha",
            rating=5,
            review_text="Great job, very pleased!",
            sentiment="positive",
            priority="Low",
            suggested_response="Thank you for your feedback!",
            db_path=self.db_path
        )

        approval = approve_review_response(
            review_id=saved["id"],
            business_id="biz-alpha",
            employee_user_id=self.user_a["id"],
            approved_response="Thank you for your wonderful review! We value your business.",
            db_path=self.db_path
        )
        self.assertEqual(approval["response_status"], "approved")
        self.assertEqual(approval["approved_by"], self.user_a["id"])

        # Check DB persistence
        reviews = get_reviews_by_business("biz-alpha", self.db_path)
        self.assertEqual(len(reviews), 1)
        rev = reviews[0]
        self.assertEqual(rev["response_status"], "approved")
        self.assertEqual(rev["approved_response"], "Thank you for your wonderful review! We value your business.")
        self.assertEqual(rev["approved_by"], self.user_a["id"])
        self.assertEqual(rev["approved_by_username"], "alice")
        self.assertIsNotNone(rev["approved_at"])

    def test_empty_response_cannot_be_approved(self):
        saved = save_review(
            business_id="biz-alpha",
            rating=3,
            review_text="Okay service.",
            suggested_response="AI response",
            db_path=self.db_path
        )

        # Empty string should fail
        with self.assertRaises(ValueError):
            approve_review_response(
                review_id=saved["id"],
                business_id="biz-alpha",
                employee_user_id=self.user_a["id"],
                approved_response="",
                db_path=self.db_path
            )

        # Whitespace-only should fail
        with self.assertRaises(ValueError):
            approve_review_response(
                review_id=saved["id"],
                business_id="biz-alpha",
                employee_user_id=self.user_a["id"],
                approved_response="   \n\t  ",
                db_path=self.db_path
            )

    def test_employee_cannot_approve_another_business_review(self):
        # Review belongs to Business B
        saved_b = save_review(
            business_id="biz-beta",
            rating=1,
            review_text="Beta error.",
            suggested_response="Beta response",
            db_path=self.db_path
        )

        # Employee A tries to approve Business B's review using Employee A's ID
        with self.assertRaises(ValueError):
            approve_review_response(
                review_id=saved_b["id"],
                business_id="biz-alpha",
                employee_user_id=self.user_a["id"],
                approved_response="Malicious approval attempt",
                db_path=self.db_path
            )

        # Employee A tries to spoof business_id as biz-beta with Employee A's user ID
        with self.assertRaises(ValueError):
            approve_review_response(
                review_id=saved_b["id"],
                business_id="biz-beta",
                employee_user_id=self.user_a["id"],
                approved_response="Malicious spoof attempt",
                db_path=self.db_path
            )

    def test_unauthenticated_or_invalid_user_cannot_approve(self):
        saved = save_review(
            business_id="biz-alpha",
            rating=4,
            review_text="Good product.",
            suggested_response="Thanks",
            db_path=self.db_path
        )

        with self.assertRaises(ValueError):
            approve_review_response(
                review_id=saved["id"],
                business_id="biz-alpha",
                employee_user_id=999999,  # Non-existent user
                approved_response="Unauthorized approval",
                db_path=self.db_path
            )

    def test_regeneration_does_not_overwrite_approved_response(self):
        saved = save_review(
            business_id="biz-alpha",
            rating=1,
            review_text="Awful experience.",
            suggested_response="Initial AI suggestion",
            db_path=self.db_path
        )

        # Approve it first
        approve_review_response(
            review_id=saved["id"],
            business_id="biz-alpha",
            employee_user_id=self.user_a["id"],
            approved_response="Final locked response approved by manager.",
            db_path=self.db_path
        )

        # Attempting to regenerate/overwrite an approved review must be blocked
        with self.assertRaises(ValueError):
            update_suggested_response(
                review_id=saved["id"],
                business_id="biz-alpha",
                new_suggested_response="New AI suggestion",
                db_path=self.db_path
            )

        # Verify that approved_response is still intact
        reviews = get_reviews_by_business("biz-alpha", self.db_path)
        self.assertEqual(reviews[0]["approved_response"], "Final locked response approved by manager.")

    def test_migration_preserves_existing_legacy_reviews(self):
        """
        Verify that a legacy database created without response_status columns
        survives init_db() migration without data loss.
        """
        legacy_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        legacy_db_path = legacy_db_file.name
        legacy_db_file.close()

        try:
            conn = sqlite3.connect(legacy_db_path)
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("""
                CREATE TABLE businesses (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    business_id TEXT NOT NULL,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'employee',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE TABLE reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    business_id TEXT NOT NULL,
                    customer_name TEXT,
                    rating INTEGER NOT NULL,
                    review_text TEXT NOT NULL,
                    sentiment TEXT,
                    confidence REAL,
                    topic TEXT,
                    priority TEXT,
                    suggested_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("INSERT INTO businesses (id, name) VALUES ('legacy-corp', 'Legacy Corp');")
            conn.execute("""
                INSERT INTO reviews (business_id, rating, review_text, sentiment, priority, suggested_response)
                VALUES ('legacy-corp', 4, 'Pre-existing legacy review', 'positive', 'Low', 'Legacy AI Response');
            """)
            conn.commit()
            conn.close()

            # Run init_db on the legacy database
            init_db(legacy_db_path)

            # Check that the legacy review exists and has default pending_approval status
            migrated_reviews = get_reviews_by_business("legacy-corp", legacy_db_path)
            self.assertEqual(len(migrated_reviews), 1)
            self.assertEqual(migrated_reviews[0]["review_text"], "Pre-existing legacy review")
            self.assertEqual(migrated_reviews[0]["response_status"], "pending_approval")
            self.assertEqual(migrated_reviews[0]["suggested_response"], "Legacy AI Response")
            self.assertIsNone(migrated_reviews[0]["approved_response"])
        finally:
            if os.path.exists(legacy_db_path):
                try:
                    os.remove(legacy_db_path)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()

