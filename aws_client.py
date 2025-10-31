"""
AWS Bedrock Client for Vision Language Classification

This module provides a client interface for AWS Bedrock foundation models,
enabling image classification using Amazon's Converse API.

Author: Edward Wong (edward_wong@eagle.co.nz)
Last Modified: 31st October 2025
"""

import json
import boto3
from typing import Dict, Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from a model text response.

    Handles various response formats including:
    - Direct JSON objects
    - Code-fenced blocks (```json\n{...}\n```)
    - Mixed text with embedded JSON
    - Multi-line JSON structures

    Args:
        text: Raw text response from the model

    Returns:
        Parsed JSON as a Python dictionary

    Raises:
        ValueError: If no valid JSON object found in text
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
    """
    AWS Bedrock client for vision language classification.

    This client uses the AWS Bedrock Converse API to send images and prompts
    to foundation models for classification tasks. Authentication is handled
    via boto3 with AWS SigV4 signing.

    Attributes:
        model_id: AWS Bedrock model identifier (e.g., "us.amazon.nova-premier-v1:0")
        region_name: AWS region (e.g., "us-east-1")
        aws_access_key_id: AWS IAM access key
        aws_secret_access_key: AWS IAM secret key
        client: boto3 bedrock-runtime client for inference
    """

    def __init__(
        self,
        model_id: str,
        region_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for AWS Bedrock. Install it with: pip install boto3"
            )

        self.model_id = model_id
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

        # Initialize boto3 bedrock-runtime client
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )

    def classify_image(
        self, encoded_image: str, sys_prompt: str, width: int, height: int
    ) -> Dict[str, Any]:
        """
        Classify an image using AWS Bedrock Converse API.

        Sends a base64-encoded image along with a system prompt to the configured
        AWS Bedrock foundation model. The model returns a JSON response with
        classification results.

        Args:
            encoded_image: Base64 encoded JPEG image string
            sys_prompt: System prompt containing classification instructions and classes
            width: Image width in pixels (for metadata)
            height: Image height in pixels (for metadata)

        Returns:
            Dictionary containing:
                - Classification: The predicted class label
                - Reason: Explanation for the classification
            Returns {"Classification": "Unknown", "Reason": "..."} on error

        Inference Configuration:
            - temperature: 0.2 (deterministic, focused responses)
            - maxTokens: 512 (sufficient for JSON classification output)
        """
        try:
            # Decode base64 image to bytes for boto3
            import base64

            image_bytes = base64.b64decode(encoded_image)

            # Call AWS Bedrock Converse API using boto3
            response = self.client.converse(
                modelId=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"text": sys_prompt},
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {"bytes": image_bytes},
                                }
                            },
                        ],
                    }
                ],
                inferenceConfig={"maxTokens": 512, "temperature": 0.2},
            )

            # Extract text from response: output.message.content[0].text
            output_text = None
            try:
                output_content = (
                    response.get("output", {}).get("message", {}).get("content", [])
                )
                if output_content and isinstance(output_content, list):
                    # Find first text content
                    for part in output_content:
                        if "text" in part:
                            output_text = part.get("text")
                            break
            except Exception:
                output_text = None

            if not output_text:
                return {"Classification": "Unknown", "Reason": "No output text"}

            # Parse JSON from model response
            try:
                result = _extract_json_from_text(output_text)
                return result
            except Exception:
                return {"Classification": "Unknown", "Reason": "JSON parse failed"}

        except ClientError:
            return {"Classification": "Unknown", "Reason": "AWS ClientError"}
        except Exception:
            return {"Classification": "Unknown", "Reason": "Exception"}
