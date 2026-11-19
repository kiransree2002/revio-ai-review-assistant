"""
Unit and integration tests for Phase 1: Authentication and Multi-Tenant Database.
Verifies password hashing, foreign keys, business isolation, review persistence,
and demo seeding using an isolated in-memory or temporary database.
"""

import os
import tempfile
import unittest
import sqlite3

from src.auth import hash_password, verify_password
from src.database import (
    init_db,
    create_business,
    get_business,
    list_businesses,
    create_user,
    authenticate_user,
    save_review,
    get_reviews_by_business,
    seed_demo_data
)


class TestAuth(unittest.TestCase):
    def test_hash_format_and_verification(self):
        password = "SecurePassword123!"
        hashed = hash_password(password)
        
        self.assertTrue(hashed.startswith("pbkdf2:sha256$200000$"))
        self.assertTrue(verify_password(password, hashed))
        self.assertFalse(verify_password("WrongPassword", hashed))
        self.assertFalse(verify_password("", hashed))
        self.assertFalse(verify_password(password, ""))

    def test_unique_salts(self):
        password = "IdenticalPassword"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        self.assertNotEqual(hash1, hash2, "Hashes must differ due to cryptographically random salts.")
        self.assertTrue(verify_password(password, hash1))
        self.assertTrue(verify_password(password, hash2))

    def test_empty_password_raises(self):
        with self.assertRaises(ValueError):
            hash_password("")


class TestDatabaseTenancy(unittest.TestCase):
    def setUp(self):
        # Create a temporary file database for clean isolated testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_db(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass

    def test_business_creation_and_retrieval(self):
        biz = create_business("acme-store", "Acme Store Inc.", self.db_path)
        self.assertEqual(biz["id"], "acme-store")
        self.assertEqual(biz["name"], "Acme Store Inc.")

        retrieved = get_business("acme-store", self.db_path)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["name"], "Acme Store Inc.")

        # Duplicate business ID should fail
        with self.assertRaises(ValueError):
            create_business("acme-store", "Acme Store Duplicate", self.db_path)

    def test_employee_creation_and_authentication(self):
        create_business("tech-corp", "Tech Corp", self.db_path)

        user = create_user("tech-corp", "alice", "SecretPass!456", role="manager", db_path=self.db_path)
        self.assertEqual(user["username"], "alice")
        self.assertEqual(user["business_id"], "tech-corp")

        # Authenticate with correct credentials
        auth = authenticate_user("alice", "SecretPass!456", self.db_path)
        self.assertIsNotNone(auth)
        self.assertEqual(auth["business_id"], "tech-corp")
        self.assertEqual(auth["business_name"], "Tech Corp")

        # Wrong password
        self.assertIsNone(authenticate_user("alice", "WrongPass", self.db_path))

        # Unknown username
        self.assertIsNone(authenticate_user("nonexistent", "SecretPass!456", self.db_path))

        # Duplicate username should fail
        with self.assertRaises(ValueError):
            create_user("tech-corp", "alice", "AnotherPass", db_path=self.db_path)

    def test_foreign_key_enforcement_for_users(self):
        # Creating user for a non-existent business must fail
        with self.assertRaises(ValueError):
            create_user("ghost-corp", "bob", "Pass1234", db_path=self.db_path)

    def test_foreign_key_enforcement_for_reviews(self):
        # Saving review for a non-existent business must fail
        with self.assertRaises(ValueError):
            save_review(
                business_id="ghost-corp",
                rating=5,
                review_text="Great experience",
                db_path=self.db_path
            )

    def test_rating_bounds_validation(self):
        create_business("store-1", "Store One", self.db_path)
        
        # Rating 0 or 6 must fail validation
        with self.assertRaises(ValueError):
            save_review("store-1", rating=0, review_text="Bad rating", db_path=self.db_path)
            
        with self.assertRaises(ValueError):
            save_review("store-1", rating=6, review_text="Invalid rating", db_path=self.db_path)

    def test_business_data_isolation(self):
        # Create two distinct businesses
        create_business("biz-alpha", "Alpha Company", self.db_path)
        create_business("biz-beta", "Beta Company", self.db_path)

        # Save reviews for Alpha
        save_review("biz-alpha", 5, "Alpha is amazing", customer_name="Alice", priority="Low", db_path=self.db_path)
        save_review("biz-alpha", 4, "Good service at Alpha", customer_name=None, priority="Medium", db_path=self.db_path)

        # Save reviews for Beta
        save_review("biz-beta", 1, "Beta was terrible", customer_name="Charlie", priority="High", db_path=self.db_path)

        # Query reviews for Alpha
        alpha_reviews = get_reviews_by_business("biz-alpha", self.db_path)
        self.assertEqual(len(alpha_reviews), 2)
        for rev in alpha_reviews:
            self.assertEqual(rev["business_id"], "biz-alpha")
            self.assertIn("Alpha", rev["review_text"])

        # Query reviews for Beta
        beta_reviews = get_reviews_by_business("biz-beta", self.db_path)
        self.assertEqual(len(beta_reviews), 1)
        self.assertEqual(beta_reviews[0]["business_id"], "biz-beta")
        self.assertEqual(beta_reviews[0]["customer_name"], "Charlie")
        self.assertEqual(beta_reviews[0]["priority"], "High")

    def test_seed_demo_data(self):
        seed_demo_data(self.db_path)
        
        demo_biz = get_business("demo-corp", self.db_path)
        self.assertIsNotNone(demo_biz)
        self.assertEqual(demo_biz["name"], "Demo Retail Corp")

        # Test demo employee login
        auth = authenticate_user("demo_employee", "demo1234", self.db_path)
        self.assertIsNotNone(auth)
        self.assertEqual(auth["business_id"], "demo-corp")


if __name__ == "__main__":
    unittest.main()
