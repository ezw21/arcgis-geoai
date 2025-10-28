import json
import requests
from typing import Dict, Any, Optional


def _append_message_log(text: str) -> None:
    """Log messages to file for debugging"""
    try:
        import os

        base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        messages_output_path = os.path.join(base_dir, "messages_output.txt")
        with open(messages_output_path, "a", encoding="utf-8") as f:
            f.write(str(text) + "\n")
    except Exception:
        pass


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

            _append_message_log(
                json.dumps(
                    {
                        "aws_preflight_request": {
                            "url": self.base_url,
                            "region": self.region_name,
                            "model_id": self.model_id,
                        }
                    }
                )
            )

            test_resp = requests.post(
                self.base_url, json=test_payload, headers=test_headers, timeout=30
            )
            status = test_resp.status_code
            rid = test_resp.headers.get("x-amzn-requestid") or test_resp.headers.get(
                "x-amzn-request-id"
            )

            _append_message_log(
                json.dumps(
                    {
                        "aws_preflight_response_meta": {
                            "status_code": status,
                            "request_id": rid,
                        }
                    }
                )
            )

            try:
                test_body = test_resp.json()
                _append_message_log(
                    "aws_preflight_response_body: " + json.dumps(test_body)[:4000]
                )
            except Exception:
                _append_message_log(
                    "aws_preflight_response_text: "
                    + (test_resp.text[:4000] if hasattr(test_resp, "text") else "")
                )

            test_resp.raise_for_status()
            return True

        except requests.exceptions.HTTPError as http_err:
            try:
                tr = http_err.response
                meta = {
                    "aws_preflight_http_error": str(http_err),
                    "status_code": getattr(tr, "status_code", None),
                    "request_id": (
                        tr.headers.get("x-amzn-requestid")
                        if hasattr(tr, "headers") and tr is not None
                        else None
                    ),
                }
                _append_message_log(json.dumps(meta))
                if tr is not None:
                    try:
                        _append_message_log(
                            "aws_preflight_error_body: " + tr.text[:4000]
                        )
                    except Exception:
                        pass
            except Exception:
                pass
            return False
        except Exception as e:
            try:
                import traceback

                _append_message_log(
                    json.dumps(
                        {
                            "aws_preflight_exception": {
                                "message": str(e),
                                "trace": traceback.format_exc(),
                                "region": self.region_name,
                                "model_id": self.model_id,
                            }
                        }
                    )
                )
            except Exception:
                pass
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

            # Log request metadata (no secrets)
            try:
                _append_message_log(
                    json.dumps(
                        {
                            "aws_request": {
                                "url": self.base_url,
                                "region": self.region_name,
                                "model_id": self.model_id,
                                "image_dims": {
                                    "width": int(width),
                                    "height": int(height),
                                },
                                "jpeg_bytes_len": len(encoded_image)
                                * 3
                                // 4,  # Approximate original size
                                "base64_len": len(encoded_image),
                            }
                        }
                    )
                )
            except Exception:
                pass

            resp = requests.post(
                self.base_url, json=payload, headers=headers, timeout=60
            )
            resp.raise_for_status()
            resp_json = resp.json()

            try:
                # Log the full Bedrock response body (truncated)
                _append_message_log(
                    "aws_response_body: " + json.dumps(resp_json)[:4000]
                )
            except Exception:
                pass

            try:
                request_id = resp.headers.get("x-amzn-requestid") or resp.headers.get(
                    "x-amzn-request-id"
                )
                _append_message_log(
                    json.dumps(
                        {
                            "aws_response_meta": {
                                "status_code": resp.status_code,
                                "request_id": request_id,
                            }
                        }
                    )
                )
            except Exception:
                pass

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
                try:
                    _append_message_log(
                        "json_parse_error: "
                        + str(e)
                        + " from text: "
                        + (output_text[:4000] if output_text else "")
                    )
                except Exception:
                    pass
                return {"Classification": "Unknown", "Reason": "Unknown"}

        except requests.exceptions.HTTPError as http_err:
            # Detailed HTTP error from AWS, capture response body
            try:
                resp = http_err.response
                body = None
                try:
                    body = resp.text
                except Exception:
                    body = None
                meta = {
                    "aws_http_error": str(http_err),
                    "status_code": getattr(resp, "status_code", None),
                    "request_id": (
                        resp.headers.get("x-amzn-requestid")
                        if hasattr(resp, "headers") and resp is not None
                        else None
                    ),
                }
                _append_message_log(json.dumps(meta))
                if body:
                    _append_message_log("aws_error_body: " + body[:4000])
            except Exception:
                pass
            return {"Classification": "Unknown", "Reason": "Unknown"}
        except Exception:
            import traceback

            try:
                error_msg = traceback.format_exc()
                _append_message_log(
                    json.dumps(
                        {
                            "aws_request_exception": {
                                "message": str(error_msg),
                                "region": self.region_name,
                                "model_id": self.model_id,
                            }
                        }
                    )
                )
            except Exception:
                # Don't let logging failures affect the main flow
                pass
            return {"Classification": "Unknown", "Reason": "Unknown"}
