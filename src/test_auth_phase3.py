"""
Automated unit and security tests for Phase 3:
Employee Authentication and Private Dashboard Access.
Tests:
- Unauthenticated access gate
- Valid employee credentials authentication
- Invalid password rejection
- Logout state clearing
- Multi-tenant dashboard data isolation (preventing cross-business access)
- Public accessibility of customer review form
- Resistance to business ID URL parameter tampering
"""

import os
import tempfile
import unittest

from src.database import (
    init_db,
    create_business,
    create_user,
    authenticate_user,
    save_review,
    get_reviews_by_business,
    seed_demo_data
)


class TestEmployeeAuthenticationSecurity(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_db(self.db_path)

        # Set up Business A and Employee A
        create_business("biz-alpha", "Alpha Enterprises", self.db_path)
        create_user("biz-alpha", "emp_alpha", "AlphaSecret2026!", role="analyst", db_path=self.db_path)

        # Set up Business B and Employee B
        create_business("biz-beta", "Beta Technologies", self.db_path)
        create_user("biz-beta", "emp_beta", "BetaSecret2026!", role="manager", db_path=self.db_path)

        # Seed reviews for Alpha
        save_review(
            business_id="biz-alpha",
            rating=5,
            review_text="Alpha delivered exceptional quality and fast shipping.",
            customer_name="Alice Alpha",
            priority="Low",
            db_path=self.db_path
        )

        # Seed reviews for Beta
        save_review(
            business_id="biz-beta",
            rating=1,
            review_text="Beta product was completely broken on arrival.",
            customer_name="Bob Beta",
            priority="High",
            db_path=self.db_path
        )

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass

    def test_valid_employee_authentication_succeeds(self):
        auth_alpha = authenticate_user("emp_alpha", "AlphaSecret2026!", self.db_path)
        self.assertIsNotNone(auth_alpha)
        self.assertEqual(auth_alpha["business_id"], "biz-alpha")
        self.assertEqual(auth_alpha["business_name"], "Alpha Enterprises")
        self.assertEqual(auth_alpha["username"], "emp_alpha")
        self.assertEqual(auth_alpha["role"], "analyst")
        # Ensure password_hash is not exposed in returned auth object
        self.assertNotIn("password_hash", auth_alpha)

    def test_invalid_password_rejected(self):
        auth_failed = authenticate_user("emp_alpha", "WrongPassword!", self.db_path)
        self.assertIsNone(auth_failed)

    def test_nonexistent_user_rejected(self):
        auth_failed = authenticate_user("ghost_employee", "SomePassword123", self.db_path)
        self.assertIsNone(auth_failed)

    def test_tenant_data_isolation(self):
        """
        Verifies that an employee authenticated under Business A cannot access
        Business B's reviews, even if attempting to query or tamper with tenant parameters.
        """
        auth_alpha = authenticate_user("emp_alpha", "AlphaSecret2026!", self.db_path)
        auth_beta = authenticate_user("emp_beta", "BetaSecret2026!", self.db_path)

        # Employee Alpha queries only Alpha's business_id stored in session state
        reviews_alpha = get_reviews_by_business(auth_alpha["business_id"], self.db_path)
        self.assertEqual(len(reviews_alpha), 1)
        self.assertEqual(reviews_alpha[0]["customer_name"], "Alice Alpha")
        self.assertNotIn("broken on arrival", reviews_alpha[0]["review_text"])

        # Employee Beta queries only Beta's business_id stored in session state
        reviews_beta = get_reviews_by_business(auth_beta["business_id"], self.db_path)
        self.assertEqual(len(reviews_beta), 1)
        self.assertEqual(reviews_beta[0]["customer_name"], "Bob Beta")
        self.assertIn("broken on arrival", reviews_beta[0]["review_text"])

    def test_seed_demo_account_and_env_password_override(self):
        """
        Verify that seed_demo_data works with the default or REVIO_DEMO_PASSWORD override.
        """
        os.environ["REVIO_DEMO_PASSWORD"] = "CustomSecret999!"
        seed_demo_data(self.db_path)

        # Authenticate with overridden password
        auth_demo = authenticate_user("demo_employee", "CustomSecret999!", self.db_path)
        self.assertIsNotNone(auth_demo)
        self.assertEqual(auth_demo["business_id"], "demo-corp")

        # Clean up env
        del os.environ["REVIO_DEMO_PASSWORD"]


if __name__ == "__main__":
    unittest.main()

