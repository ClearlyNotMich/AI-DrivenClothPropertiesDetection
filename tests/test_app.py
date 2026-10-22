"""Regression checks for the local metadata demo, without ML dependencies."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

APP_PATH = Path(__file__).resolve().parents[1] / "code" / "app.py"
SPEC = importlib.util.spec_from_file_location("fabric_demo", APP_PATH)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


class FabricDemoTests(unittest.TestCase):
    def setUp(self):
        self.client = demo.app.test_client()

    def test_index_and_existing_placeholder_are_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"does not classify images or run model inference", response.data)
        self.assertIn(b'fetch("/api/fabrics")', response.data)
        self.assertIn(b'src="/placeholder.jpg"', response.data)
        with self.client.get("/placeholder.jpg") as image:
            self.assertEqual(image.status_code, 200)
            self.assertEqual(image.mimetype, "image/jpeg")
            self.assertTrue(image.data.startswith(b"\xff\xd8"))

    def test_real_csv_preserves_identifiers_and_quoted_composition(self):
        response = self.client.get("/api/fabrics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data["fields"]), 44)
        self.assertEqual(len(data["fabrics"]), 10)
        record = data["fabrics"]["22F00002"]
        self.assertEqual(record["Textile Description"], "Wool (CK)")
        self.assertEqual(record["Composition"], "Wo (warp), PES, CV, EL (Weft)")
        self.assertEqual(record["Structure and Construction"], "Woven")
        self.assertEqual(len(record), 44)

    def test_legacy_encoding_and_quoted_multiline_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "fabric.csv"
            fixture.write_text('General Identifier, Description,Composition\n21F00012,"Café\ncloth","cotton, silk"\n', encoding="ISO-8859-1")
            with patch.object(demo, "DATA_PATH", fixture):
                data = self.client.get("/api/fabrics").get_json()
            self.assertEqual(data["fabrics"]["21F00012"]["Description"], "Café\ncloth")
            self.assertEqual(data["fabrics"]["21F00012"]["Composition"], "cotton, silk")

    def test_missing_csv_returns_readable_json_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(demo, "DATA_PATH", Path(folder) / "missing.csv"):
                with self.assertLogs(demo.app.logger, level="ERROR"):
                    response = self.client.get("/api/fabrics")
        self.assertEqual(response.status_code, 503)
        self.assertIn("unavailable", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
