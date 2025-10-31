"""
Test AWS Bedrock authentication and list available models.
This script helps diagnose if issues are related to authentication or endpoint structure.
"""

import json
import sys
import os
import keyring

# Add current directory to path to import aws_client
sys.path.insert(0, os.path.dirname(__file__))

from aws_client import AWSBedrockClient

try:
    from botocore.exceptions import ClientError
except ImportError:
    ClientError = Exception


def list_foundation_models(client: AWSBedrockClient) -> list:
    """
    Lists available Amazon Bedrock foundation models.
    Uses the bedrock client (control plane) to verify authentication.

    Returns:
        List of model summaries if successful, empty list if error
    """
    try:
        response = client.bedrock_client.list_foundation_models()
        models = response.get("modelSummaries", [])
        return models
    except ClientError:
        return []
    except Exception:
        return []


def test_connection(client: AWSBedrockClient) -> bool:
    """Test AWS Bedrock connection with a simple text message"""
    try:
        response = client.client.converse(
            modelId=client.model_id,
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            inferenceConfig={"maxTokens": 64, "temperature": 0.0},
        )
        return True
    except ClientError:
        return False
    except Exception:
        return False


def main():
    """Test AWS authentication by listing available models"""

    # Get connection file from command line or use default
    import sys

    if len(sys.argv) > 1:
        connection_file = sys.argv[1]
    else:
        connection_file = "AWS_T6.ais"  # Default connection file

    # Read credentials from connection file
    try:
        connection_path = os.path.join(os.path.dirname(__file__), connection_file)
        if not os.path.exists(connection_path):
            print(f"Error: Connection file not found: {connection_path}")
            return

        with open(connection_path, "r") as f:
            config = json.load(f)

        model_id = config["serviceProviderProperties"]["model_id"]
        region_name = config["serviceProviderProperties"]["aws_region_name"]
        aws_access_key = config["serviceProviderProperties"]["aws_access_key"]

        # Retrieve secret key from Windows Credential Manager using keyring
        # Just like how other AI service providers do it
        aws_secret_key = config["authenticationSecrets"]["token"]
        if (
            "authenticationProperties" in config
            and "parameterName" in config["authenticationProperties"]
        ):
            param_name = config["authenticationProperties"]["parameterName"]
            uuid = config["authenticationSecrets"]["token"]
            credential = keyring.get_password(uuid, param_name)
            if credential is not None:
                aws_secret_key = credential
                print(f"✓ Retrieved secret key from Windows Credential Manager")
            else:
                print(f"⚠ WARNING: Could not retrieve credential from keyring")
                print(f"  UUID: {uuid}")
                print(f"  Parameter: {param_name}")
                print(f"  Falling back to raw token value")

        print(f"Configuration loaded:")
        print(f"  Model ID: {model_id}")
        print(f"  Region: {region_name}")
        print(f"  Access Key: {aws_access_key}")
        print(f"  Secret Key: {'*' * 20}{aws_secret_key[-4:]}")
        print()

    except Exception as e:
        print(f"Error reading connection file: {e}")
        return

    # Create AWS client
    try:
        print("Creating AWS Bedrock client...")
        client = AWSBedrockClient(
            model_id=model_id,
            region_name=region_name,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )
        print("✓ Client created successfully")
        print()
    except Exception as e:
        print(f"✗ Error creating client: {e}")
        return

    # Test 1: List available models (tests authentication with control plane)
    print("=" * 70)
    print("TEST 1: List Foundation Models (Control Plane)")
    print("=" * 70)
    try:
        models = list_foundation_models(client)
        if models:
            print(f"✓ Successfully listed {len(models)} models")
            print("\nAvailable models in your region:")
            for model in models:
                model_name = model.get("modelName", "Unknown")
                model_id = model.get("modelId", "Unknown")
                input_modalities = model.get("inputModalities", [])
                output_modalities = model.get("outputModalities", [])
                print(f"  - {model_name}")
                print(f"    ID: {model_id}")
                print(f"    Input: {', '.join(input_modalities)}")
                print(f"    Output: {', '.join(output_modalities)}")
                print()
        else:
            print("✗ No models returned")
    except Exception as e:
        print(f"✗ Error listing models: {e}")

    print()

    # Test 2: Test connection with simple text message (tests inference endpoint)
    print("=" * 70)
    print("TEST 2: Test Connection (Data Plane - bedrock-runtime)")
    print("=" * 70)
    try:
        success = test_connection(client)
        if success:
            print(f"✓ Successfully connected to model: {model_id}")
        else:
            print(f"✗ Connection failed")
    except Exception as e:
        print(f"✗ Error testing connection: {e}")


if __name__ == "__main__":
    main()
