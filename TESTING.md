# AWS Bedrock Integration - Testing Guide

## Overview

This directory contains tools for testing AWS Bedrock integration with the VisionLanguageClassification raster function.

## Test Files

### Main Test Script

**`test_aws_auth.py`** - Primary authentication and functionality test

- Tests connection to AWS Bedrock
- Lists available models
- Validates credentials work correctly
- Uses Windows Credential Manager for secure credential storage

**Usage:**

```bash
python test_aws_auth.py [connection_file.ais]
```

Default connection file: `AWS_T6.ais`

Example:

```bash
python test_aws_auth.py
python test_aws_auth.py my_aws_config.ais
```

## Connection File Format

AWS connection files (`.ais`) use this structure:

```json
{
  "version": "1.0",
  "serviceProvider": "AWS",
  "protocol": "",
  "host": "",
  "authenticationScheme": "accessToken",
  "authenticationProperties": {
    "parameterType": "header",
    "parameterName": "aws_secret_key"
  },
  "authenticationSecrets": {
    "token": "uuid-from-credential-manager"
  },
  "serviceProviderProperties": {
    "model_id": "us.amazon.nova-premier-v1:0",
    "aws_region_name": "us-east-1",
    "aws_access_key": "AKIA..."
  }
}
```

**Important:**

- `.ais` files are in `.gitignore` - they contain sensitive access keys
- The `token` field in `authenticationSecrets` is a UUID used to look up the actual secret key in Windows Credential Manager
- Never commit `.ais` files to git

## Credential Management

### How it works:

1. AWS Access Key ID is stored in the `.ais` file (public identifier)
2. AWS Secret Access Key is stored in Windows Credential Manager (secure)
3. The `.ais` file contains a UUID that maps to the Credential Manager entry

### To update credentials:

1. Update the secret key in Windows Credential Manager manually
2. Or create a new access key pair in AWS IAM Console
3. Update the access key in your `.ais` file if needed

### To verify credentials:

1. Run `python test_aws_auth.py`
2. Check that both tests pass (model listing and connection test)

## Required IAM Permissions

Your AWS IAM user needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:ListFoundationModels", "bedrock:InvokeModel"],
      "Resource": "*"
    }
  ]
}
```

## Troubleshooting

### "InvalidClientTokenId" error

- The access key doesn't exist or is inactive
- Check AWS IAM Console to verify the key is Active

### "SignatureDoesNotMatch" error

- The secret key doesn't match the access key
- Verify the secret key in Credential Manager is correct
- The access key and secret key must be from the same key pair

### "AccessDeniedException" error

- The IAM user lacks required Bedrock permissions
- Add `bedrock:InvokeModel` and `bedrock:ListFoundationModels` permissions

### Connection file not found

- Specify the full path to the connection file
- Or place it in the same directory as the test script

## Test Output

The test script displays results directly to the console:

- Lists all available Bedrock models in your region
- Shows connection test results
- No log files are created

## Security Best Practices

1. ✅ **DO** store `.ais` files locally only (already in `.gitignore`)
2. ✅ **DO** use Windows Credential Manager for secret keys
3. ✅ **DO** rotate credentials regularly
4. ✅ **DO** use IAM users with minimal required permissions
5. ❌ **DON'T** commit `.ais` files to git
6. ❌ **DON'T** hardcode credentials in Python files

## File Structure

```
├── test_aws_auth.py          # Main test script
├── aws_client.py             # AWS Bedrock client implementation
├── VisionLanguageClassification.py  # Main raster function
├── AWS_T6.ais               # Connection file (in .gitignore)
└── TESTING.md              # This file
```

## Integration with ArcGIS

The main raster function (`VisionLanguageClassification.py`) uses the same credential management:

1. Reads connection file specified in the geoprocessing tool
2. Extracts UUID from `authenticationSecrets.token`
3. Retrieves actual secret key from Windows Credential Manager using `keyring`
4. Initializes AWS Bedrock client with credentials

This ensures credentials are never hardcoded and are stored securely.
