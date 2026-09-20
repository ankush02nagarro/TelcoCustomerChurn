"""With the server running: python test_api.py (standard library only)."""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def post(payload):
    request = Request("http://127.0.0.1:8000/predict",
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=30) as response:
        return json.load(response)


if __name__ == "__main__":
    payload = json.loads((ROOT / "sample_request.json").read_text())
    result = post(payload)
    assert result["prediction"] in ["Yes", "No"]
    assert 0 <= result["churn_probability"] <= 1
    (ROOT / "sample_response.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Actual API response:", json.dumps(result, indent=2))
    for invalid in [{}, {**payload, "tenure": -1}, {**payload, "Contract": "Invalid"}]:
        try:
            post(invalid)
        except HTTPError as exc:
            assert exc.code == 422, f"Expected 422; got {exc.code}"
        else:
            raise AssertionError("Invalid input should have been rejected")
    print("Valid request and three invalid-input checks passed.")
