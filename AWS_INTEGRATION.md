# AWS Bedrock Integration for VisionLanguageClassification

This document explains the AWS Bedrock support added to the VisionLanguageClassification raster function for ArcGIS.

## Overview

AWS Bedrock integration enables the raster function to use Amazon's foundation models (like Nova, Claude, etc.) for image classification alongside existing Azure OpenAI and OpenAI providers.

## Architecture

### Core Components

1. **aws_client.py** - AWS Bedrock client implementation
2. **VisionLanguageClassification.py** - Main raster function with AWS integration
3. **test_aws_auth.py** - Authentication and connectivity testing utility

---

## 1. AWS Client (`aws_client.py`)

### Purpose

Handles all AWS Bedrock API interactions using the boto3 SDK with proper AWS SigV4 authentication.

### Key Features

#### Authentication

- Uses boto3 SDK for AWS authentication
- Supports AWS access key + secret key credentials
- Implements AWS Signature Version 4 (SigV4) signing automatically via boto3

#### Main Method: `classify_image()`

```python
def classify_image(self, encoded_image: str, sys_prompt: str, width: int, height: int) -> Dict[str, Any]
```

**What it does:**

1. Decodes base64-encoded JPEG image to bytes
2. Calls AWS Bedrock Converse API with image and system prompt
3. Extracts text response from model output
4. Parses JSON classification result
5. Returns dictionary with `Classification` and `Reason` keys

**API Details:**

- **Endpoint**: AWS Bedrock Runtime (`bedrock-runtime` service)
- **API**: Converse API
- **Input**: System prompt + JPEG image bytes
- **Output**: JSON structured response

#### Helper Method: `_extract_json_from_text()`

Robust JSON parser that handles:

- Direct JSON objects
- Code-fenced responses (`json ... `)
- Mixed text with embedded JSON
- Multi-line JSON structures

---

## 2. Main Integration (`VisionLanguageClassification.py`)

### Three Key Code Blocks

#### Block 1: Configuration Parsing (Lines ~346-369)

```python
# NEW AWS support - read AWS-specific connection parameters
if service_provider == "AWS":
    if "model_id" in service_provider_props:
        self.deployment_name = service_provider_props["model_id"]
    if "aws_region_name" in service_provider_props:
        self.aws_region_name = service_provider_props["aws_region_name"]
    if "aws_access_key" in service_provider_props:
        self.aws_access_key_id = service_provider_props["aws_access_key"]
```

**Purpose:** Extracts AWS-specific configuration from connection file (.ais)

**Parameters read:**

- `model_id`: AWS Bedrock model identifier (e.g., "us.amazon.nova-premier-v1:0")
- `aws_region_name`: AWS region (e.g., "us-east-1")
- `aws_access_key`: AWS IAM access key ID

**Secret Key Retrieval:**

- Secret key is stored in Windows Credential Manager
- Retrieved via `keyring.get_password(uuid, param_name)`
- UUID comes from `authenticationSecrets.token` in connection file
- Same pattern used by other AI providers (Azure, OpenAI)

#### Block 2: Client Initialization (Lines ~408-417)

```python
# NEW AWS support
elif service_provider == "AWS":
    from aws_client import AWSBedrockClient
    self.client = AWSBedrockClient(
        model_id=self.deployment_name,
        region_name=self.aws_region_name,
        aws_access_key_id=self.aws_access_key_id,
        aws_secret_access_key=self.api_key,
    )
```

**Purpose:** Creates AWS Bedrock client instance with credentials

**Note:** `self.api_key` contains the secret key (already retrieved from keyring in earlier code)

#### Block 3: Inference Routing (Lines ~473-476)

```python
# NEW AWS support - route to AWS client
if hasattr(self.client, "classify_image"):
    answer = self.client.classify_image(
        encoded_image, sys_prompt, width, height
    )
```

**Purpose:** Routes AWS requests to the dedicated `classify_image()` method

**Why this works:**

- Azure/OpenAI clients use standard `chat.completions.create()`
- AWS client has custom `classify_image()` method
- `hasattr()` check distinguishes between provider types

---

## 3. Inference Configuration

### Temperature Setting

Controls the randomness/creativity of model responses:

| Value   | Behavior               | Use Case                                  |
| ------- | ---------------------- | ----------------------------------------- |
| **0.2** | Deterministic, focused | ✅ **Classification tasks** (AWS default) |
| 1.0     | Random, creative       | Creative writing, brainstorming           |

**AWS Implementation:**

```python
inferenceConfig={"maxTokens": 512, "temperature": 0.2}
```

**Why 0.2 for classification:**

- Produces consistent results for same input
- Picks most likely tokens (highest probability)
- Reduces hallucination and randomness
- Better for structured JSON output

### Max Tokens Setting

Limits the **OUTPUT** length (model's response), not input.

**AWS Setting: 512 tokens**

**Why 512 is appropriate:**

- Classification responses are typically 50-150 tokens:
  ```json
  {
    "Classification": "Building",
    "Reason": "Rectangular structure with visible roof"
  }
  ```
- 512 provides comfortable buffer
- Prevents runaway responses
- Reduces API costs (charged per token)

**Comparison:**

- AWS: 512 tokens (explicit limit) ✅
- Azure/OpenAI: ~4096 tokens (model default, no limit set) ⚠️

---

## 4. Connection File Format (.ais)

AWS connection files follow the versioned format (version 1.0):

```json
{
  "version": "1.0",
  "serviceProviderProperties": {
    "model_id": "us.amazon.nova-premier-v1:0",
    "aws_region_name": "us-east-1",
    "aws_access_key": "AKIAXI5PHSI2XXXXXXXX"
  },
  "authenticationSecrets": {
    "token": "3bbbf185-b5d3-11f0-9609-f077c3977033"
  },
  "authenticationProperties": {
    "parameterName": "aws_secret_key"
  }
}
```

**Security Notes:**

- Access key stored in file (less sensitive)
- Secret key stored in Windows Credential Manager (secure)
- Retrieved via UUID token at runtime
- Connection files excluded from git (.gitignore)

---

## 5. Testing Utility (`test_aws_auth.py`)

### Purpose

Validates AWS Bedrock authentication and lists available models.

### Usage

```bash
python test_aws_auth.py [connection_file.ais]
```

**Default:** Uses `AWS_T6.ais` if no file specified

### What it tests

#### Test 1: List Foundation Models

- Uses AWS Bedrock control plane (`bedrock` service)
- Lists all available foundation models in your region
- Validates IAM permissions
- Shows model capabilities (input/output modalities)

#### Test 2: Test Connection

- Uses AWS Bedrock data plane (`bedrock-runtime` service)
- Sends simple "Hi" message via Converse API
- Validates inference endpoint connectivity
- Confirms model is accessible

### Test Functions

```python
def list_foundation_models(client: AWSBedrockClient) -> list
def test_connection(client: AWSBedrockClient) -> bool
```

**Note:** These helper functions are in the test file, NOT in production code (`aws_client.py`).

---

## Key Design Decisions

### 1. boto3 SDK vs Direct HTTP

- **Chosen:** boto3 SDK
- **Why:** Handles AWS SigV4 authentication automatically, official AWS SDK, more robust

### 2. Converse API vs InvokeModel

- **Chosen:** Converse API
- **Why:** Standardized interface across models, simpler message format, better for multi-modal input

### 3. Temperature 0.2

- **Chosen:** Low temperature for deterministic results
- **Why:** Classification requires consistency, not creativity

### 4. Separate aws_client.py

- **Chosen:** Dedicated module for AWS logic
- **Why:** Minimal changes to main raster function, easier testing, cleaner separation

### 5. Test Functions Location

- **Chosen:** In test file, not production code
- **Why:** Production code only needs `classify_image()`, keeps aws_client.py minimal

---

## Dependencies

Required Python packages:

```
boto3>=1.28.0
botocore>=1.31.0
keyring>=23.0.0
```

Install via:

```bash
pip install boto3 botocore keyring
```

---

## IAM Permissions Required

Your AWS IAM user/role needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*:*:foundation-model/*"
    },
    {
      "Effect": "Allow",
      "Action": ["bedrock:ListFoundationModels"],
      "Resource": "*"
    }
  ]
}
```

---

## Error Handling

All AWS errors return safe default:

```python
{"Classification": "Unknown", "Reason": "Error description"}
```

Common errors handled:

- `ClientError`: AWS API errors (auth, permissions, throttling)
- `JSONDecodeError`: Model response parsing issues
- `ValueError`: Empty or invalid responses

---

## Comparison with Other Providers

| Feature                | AWS Bedrock                | Azure OpenAI               | OpenAI                     |
| ---------------------- | -------------------------- | -------------------------- | -------------------------- |
| **Authentication**     | Access key + Secret key    | API key                    | API key                    |
| **Credential Storage** | Windows Credential Manager | Windows Credential Manager | Windows Credential Manager |
| **SDK**                | boto3                      | openai (Azure)             | openai                     |
| **Temperature**        | 0.2 ✅                     | 1.0 (default) ⚠️           | 1.0 (default) ⚠️           |
| **Max Tokens**         | 512 ✅                     | 4096 (default) ⚠️          | 4096 (default) ⚠️          |
| **API Endpoint**       | Converse API               | Chat Completions           | Chat Completions           |
| **Image Format**       | Bytes                      | Base64 URL                 | Base64 URL                 |

---

## Files Modified/Added

### New Files

- `aws_client.py` - AWS Bedrock client
- `test_aws_auth.py` - Testing utility
- `AWS_INTEGRATION.md` - This documentation

### Modified Files

- `VisionLanguageClassification.py` - Added 3 AWS code blocks
- `.gitignore` - Added `*.ais` exclusion

### Connection Files (Not in Git)

- `AWS_T6.ais` - Active connection file with credentials

---

## Future Improvements

Potential enhancements:

1. Add `temperature` and `max_tokens` to Azure/OpenAI for consistency
2. Support streaming responses for real-time feedback
3. Add retry logic with exponential backoff
4. Support for additional AWS regions
5. Model capability detection (vision vs text-only)

---

## Troubleshooting

### "InvalidSignatureException"

- Secret key doesn't match access key
- Check Windows Credential Manager entries
- Verify UUID in connection file matches keyring entry

### "No models listed"

- Check IAM permissions (`bedrock:ListFoundationModels`)
- Verify region has Bedrock enabled
- Check AWS credentials are active

### "Classification: Unknown"

- Model may not support images
- Check model ID is correct
- Verify IAM permissions for `bedrock:InvokeModel`
- Review test_aws_auth.py output for detailed errors

---

## Testing Workflow

1. **Create connection file** with access key and region
2. **Store secret key** in Windows Credential Manager
3. **Run test script:** `python test_aws_auth.py`
4. **Verify:** 100+ models listed, connection test passes
5. **Use in ArcGIS:** Load connection file in raster function

---

## References

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Converse API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
- [boto3 Bedrock Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime.html)
