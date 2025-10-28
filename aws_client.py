import json
import requests
from typing import Dict, Any, Optional


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from a model text response.
    Handles code-fenced blocks like ```json\n{...}\n``` and multi-line content.
    Returns a Python object (dict/list) if possible, else raises.
    """
    if text is None:
        raise ValueError("empty text")

    # Try a direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Strip code fences ```json ... ``` or ``` ... ```
    import re as _re

    fence = _re.search(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", text)
    candidate = fence.group(1) if fence else text

    # Try direct parse again after stripping
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Brace matching to extract the first complete JSON object
    start = None
    depth = 0
    for idx, ch in enumerate(candidate):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    snippet = candidate[start : idx + 1]
                    try:
                        return json.loads(snippet)
                    except Exception:
                        # keep scanning in case of nested objects followed by more text
                        pass

    # Last resort: DOTALL regex
    m = _re.search(r"\{[\s\S]*\}", candidate)
    if m:
        return json.loads(m.group(0))

    raise ValueError("No JSON object found in text")


class AWSBedrockClient:
    """AWS Bedrock client for making API requests"""

    def __init__(self, model_id: str, region_name: str, api_key: Optional[str] = None):
        self.model_id = model_id
        self.region_name = region_name
        self.api_key = api_key
        self.base_url = f"https://bedrock-runtime.{region_name}.amazonaws.com/model/{model_id}/converse"

    def test_connection(self) -> bool:
        """Test AWS Bedrock connection with a simple text message"""
        try:
            test_payload = {
                "messages": [{"role": "user", "content": [{"text": "Hi How are you"}]}],
                "inferenceConfig": {"maxTokens": 64, "temperature": 0.0},
            }
            test_headers = {
                "Content-Type": "application/json",
                "Authorization": (f"Bearer {self.api_key}" if self.api_key else None),
            }
            test_headers = {k: v for k, v in test_headers.items() if v is not None}

            test_resp = requests.post(
                self.base_url, json=test_payload, headers=test_headers, timeout=30
            )
            status = test_resp.status_code
            rid = test_resp.headers.get("x-amzn-requestid") or test_resp.headers.get(
                "x-amzn-request-id"
            )

            test_resp.raise_for_status()
            return True

        except requests.exceptions.HTTPError as http_err:
            return False
        except Exception as e:
            return False

    def classify_image(
        self, encoded_image: str, sys_prompt: str, width: int, height: int
    ) -> Dict[str, Any]:
        """
        Classify an image using AWS Bedrock Converse API

        Args:
            encoded_image: Base64 encoded JPEG image
            sys_prompt: System prompt for classification
            width: Image width
            height: Image height

        Returns:
            Dictionary with Classification and Reason keys
        """
        try:
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": sys_prompt},
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {"bytes": encoded_image},
                                }
                            },
                        ],
                    }
                ],
                "inferenceConfig": {"maxTokens": 512, "temperature": 0.2},
            }
            headers = {
                "Content-Type": "application/json",
                # Using API key as bearer if provided; alternatively use SigV4 via AWS SDK.
                "Authorization": (f"Bearer {self.api_key}" if self.api_key else None),
            }
            # Remove None headers
            headers = {k: v for k, v in headers.items() if v is not None}

            resp = requests.post(
                self.base_url, json=payload, headers=headers, timeout=60
            )
            resp.raise_for_status()
            resp_json = resp.json()

            # Bedrock converse response shape: output.message.content[0].text
            output_text = None
            try:
                output_message = (
                    resp_json.get("output", {}).get("message", {}).get("content", [])
                )
                if output_message and isinstance(output_message, list):
                    # Find first text content
                    for part in output_message:
                        if "text" in part:
                            output_text = part.get("text")
                            break
            except Exception:
                output_text = None

            if not output_text:
                # Fallback: some models return at top-level or different shape
                output_text = resp_json.get("content", "") or resp_json.get(
                    "message", ""
                )

            # Expect a JSON object in output_text
            try:
                return _extract_json_from_text(output_text)
            except Exception as e:
                return {"Classification": "Unknown", "Reason": "Unknown"}

        except requests.exceptions.HTTPError as http_err:
            return {"Classification": "Unknown", "Reason": "Unknown"}
        except Exception:
            return {"Classification": "Unknown", "Reason": "Unknown"}
