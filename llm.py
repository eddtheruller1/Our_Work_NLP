"""A small Gemini client for labelling and translating headlines.

The key comes from the GEMINI_API_KEY environment variable, or from a `.env`
file next to this script (see `.env.example`). The model comes from
GEMINI_MODEL (default gemini-2.5-flash).
"""
import os
import time
from pathlib import Path

HERE = Path(__file__).parent
_env = HERE / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


class Gemini:
    def __init__(self, model: str):
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in .env, see .env.example)")
        self.client = genai.Client(api_key=key)
        self.model = model

    def generate_json(self, instruction: str, document: str, tries: int = 4):
        """Returns (answer_text, model name). Waits and retries on rate limits."""
        for attempt in range(tries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=f"{instruction}\n\n{document}",
                    config={"response_mime_type": "application/json", "temperature": 0})
                return resp.text, self.model
            except Exception:
                if attempt == tries - 1:
                    raise
                time.sleep(5 * (attempt + 1))


def build_chain(provider: str = "gemini") -> Gemini:
    return Gemini(os.getenv("GEMINI_MODEL") or "gemini-2.5-flash")
