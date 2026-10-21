import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # requirements.txt installs Streamlit in normal environments
    AppTest = None


@unittest.skipIf(AppTest is None, "Streamlit is not installed")
class DashboardAppTests(unittest.TestCase):
    def test_all_pages_render_without_exceptions(self):
        app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        self.assertFalse(app.exception)
        for page in ["Stage 1", "Backtests", "Stage 2", "Methodology"]:
            app.radio[0].set_value(page).run(timeout=30)
            self.assertFalse(app.exception, page)


if __name__ == "__main__":
    unittest.main()
