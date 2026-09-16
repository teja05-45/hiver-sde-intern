import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.api.app import app


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_returns_200(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)

    def test_health_reports_mock_mode_by_default(self):
        r = self.client.get("/health")
        data = r.get_json()
        self.assertIn("mock_mode", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")


if __name__ == "__main__":
    unittest.main()
