import importlib
from importlib import reload, import_module
import json
import os
import sys
import arcpy, math

# sys.path.append(os.path.dirname(__file__))
import numpy as np
from PIL import Image
import time
import requests
import base64
from io import BytesIO
import json
from openai import AzureOpenAI, OpenAI


def get_available_device(max_memory=0.8):
    """
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0
    """
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available


def process_llama_request(self, temp_img):
    sys_prompt1 = "You are an AI Image Classifier that helps people classify images to one of the several provided categories. Also, share the reasoning behind your classification. You must return the response as a json"

    sys_prompt = ' You are an AI Image Classifier that helps people classify images to one of the several provided categories. Also, share the reasoning behind your classification. You must return the response as a json with the following keys: 1. Classification and 2. Reason in the format {"Classification":"", "Reason":""}. The provided categories are: '

    sys_prompt = self.additional_context + sys_prompt + str(self.classes)
    messages = [
        {
            "role": sys_prompt1,
            "content": [{"type": "image"}, {"type": "text", "text": sys_prompt}],
        }
    ]
    input_text = self.processor.apply_chat_template(
        messages, add_generation_prompt=True
    )

    inputs = self.processor(
        temp_img, input_text, add_special_tokens=False, return_tensors="pt"
    ).to(self.device_id)

    output = self.model.generate(**inputs, max_new_tokens=512)

    text = self.processor.decode(output[0])
    import re, json

    try:
        json_match = re.search(r"(\{.*\})", text[len(input_text) :])
        if json_match:
            json_data = json.loads(json_match.group(1))
            answer = json_data
        else:
            answer = {"Classification": "Unknown", "Reason": "Unknown"}

        # if text.find("**Classification:**") != -1:
        # classification_start = text.find("**Classification:**") + len("**Classification:**")
        # classification_end = text.find("\n", classification_start)
        # classification = text[classification_start:classification_end].strip()

        # reason_start = text.find("**Reason:**") + len("**Reason:**")
        # reason_end = text.find("\n", reason_start)
        # if (reason_end == -1) or (reason_start == reason_end):
        # reason_end = text.find("<|eot_id|>", reason_start)

        # reason = text[reason_start:reason_end].strip()

        # Store the results in a dictionary
        # result = {
        # "Classification": classification,
        # "Reason": reason
        # }
        # answer = result
        # else:
        # answer = {'Classification':'Unknown', 'Reason':'Unknown'}
    except:
        answer = {"Classification": "Unknown", "Reason": "Unknown"}

    return answer


def _append_message_log(text: str) -> None:
    try:
        base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
        messages_output_path = os.path.join(base_dir, "messages_output.txt")
        with open(messages_output_path, "a", encoding="utf-8") as f:
            f.write(str(text) + "\n")
    except Exception:
        pass


def _extract_json_from_text(text: str):
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


features = {
    "displayFieldName": "",
    "fieldAliases": {
        "OID": "OID",
        "Reason": "Reason",
        "Confidence": "Confidence",
        "Shape": "Shape",
        "Label": "Label",
    },
    "geometryType": "esriGeometryPolygon",
    "fields": [
        {"name": "OID", "type": "esriFieldTypeOID", "alias": "OID"},
        {
            "name": "Reason",
            "type": "esriFieldTypeString",
        },
        {"name": "Confidence", "type": "esriFieldTypeDouble", "alias": "Confidence"},
        {"name": "Shape", "type": "esriFieldTypeGeometry", "alias": "Shape"},
        {"name": "Label", "type": "esriFieldTypeString", "alias": "Label"},
    ],
    "features": [],
}


class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4


class VisionLanguageClassification:
    def __init__(self):
        self.name = "Object classifier"
        self.description = "This python raster function applies deep learning model to classify objects from overlaid imagery"

    def initialize(self, **kwargs):

        if "model" not in kwargs:
            return

        # Read esri model definition (emd) file
        model = kwargs["model"]
        model_as_file = True

        try:
            with open(model, "r") as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        # sys.path.append(os.path.dirname(__file__))

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        self.device_id = None
        if "device" in kwargs:
            device = kwargs["device"]
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                # try:
                # import torch
                # except Exception:
                # raise Exception("PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                # torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
                self.device_id = device
            else:
                arcpy.env.processorType = "CPU"
                self.device_id = "cpu"

        # import torch
        import sys

        transformers_root_dir = os.path.dirname(__file__)
        if transformers_root_dir not in sys.path:
            sys.path.insert(0, transformers_root_dir)
        # self.torch = torch

    def getParameterInfo(self):

        # PRF needs values of these parameters from gp tool user,
        # either from gp tool UI or emd (a json) file.
        required_parameters = [
            {
                # To support mini batch, it is required that Classify Objects Using Deep Learning geoprocessing Tool
                # passes down a stack of raster tiles to PRF for model inference, the keyword required here is 'rasters'.
                "name": "rasters",
                "dataType": "rasters",
                "value": None,
                "required": True,
                "displayName": "Rasters",
                "description": "The collection of overlapping rasters to objects to be classified",
            },
            {
                "name": "model",
                "dataType": "string",
                "required": True,
                "displayName": "Input Model Definition (EMD) File",
                "description": "Input model definition (EMD) JSON file",
            },
            {
                "name": "device",
                "dataType": "numeric",
                "required": False,
                "displayName": "Device ID",
                "description": "Device ID",
            },
        ]

        required_parameters.extend(
            [
                {
                    "name": "classes",
                    "dataType": "string",
                    "required": True,
                    "value": "",
                    "displayName": "Classes",
                    "description": "Classes",
                }
            ]
        )

        required_parameters.extend(
            [
                {
                    "name": "additional_context",
                    "dataType": "string",
                    "required": True,
                    "value": "",
                    "displayName": "additional_context",
                    "description": "Additional context for the image",
                }
            ]
        )
        required_parameters.extend(
            [
                {
                    "name": "strict_classification",
                    "dataType": "boolean",
                    "required": True,
                    "value": "false",
                    "displayName": "strict_classification",
                    "description": "Flag to handle hallucinated classes",
                }
            ]
        )
        required_parameters.extend(
            [
                {
                    "name": "ai_connection_file",
                    "dataType": "string",
                    "required": True,
                    "value": "",
                    "displayName": "ai_connection_file",
                    "description": "Path to the ai connection json file",
                }
            ]
        )

        return required_parameters

    def getConfiguration(self, **scalars):

        if "BatchSize" not in self.json_info and "batch_size" not in scalars:
            self.batch_size = 1
        elif "BatchSize" not in self.json_info and "batch_size" in scalars:
            self.batch_size = int(scalars["batch_size"])
        else:
            self.batch_size = int(self.json_info["BatchSize"])

        self.classes = scalars.get("classes")
        self.classes_list = self.classes.strip(",").split(",")
        self.classes_list = [cls.strip() for cls in self.classes.strip(",").split(",")]
        self.additional_context = scalars.get("additional_context", "")
        self.strict_classification = scalars.get(
            "strict_classification", "false"
        ).lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]
        self.ai_connection_file = scalars.get("ai_connection_file", None)
        self.connection_token = scalars.get("token", None)
        conn_params = {}
        import json
        import importlib

        try:
            keyring = importlib.import_module(
                "keyring"
            )  # optional; used for fetching secrets from OS keyring
        except Exception:  # ImportError or environment-specific
            keyring = None

        if self.ai_connection_file:
            if os.sep not in self.ai_connection_file:
                raise Exception(
                    "Unable to access the path of the file.Note that this model is not currently supported in ArcGIS online or enterprise."
                )
            else:
                self.ai_connection_file = self.ai_connection_file.replace("\\", "/")

                with open(self.ai_connection_file) as json_file:
                    conn_params = json.load(json_file)
        else:
            raise Exception("Connection File was not provided.")

        if "version" in conn_params:
            out_dict = conn_params

            if (
                "authenticationProperties" in out_dict
                and "authenticationSecrets" in out_dict
            ):
                auth_prop = out_dict["authenticationProperties"]
                auth_secret = out_dict["authenticationSecrets"]
                if "parameterName" in auth_prop and "token" in auth_secret:
                    param_name = auth_prop["parameterName"]
                    uuid = auth_secret["token"]
                    credential = keyring.get_password(uuid, param_name)
                    if credential != None:
                        out_dict["authenticationSecrets"]["token"] = credential

            # con = out_dict
            conn_param_v = out_dict
            protocol = conn_param_v.get("protocol", None)

            self.azure_endpoint = protocol + "://" + conn_param_v.get("host", None)
            self.api_key = conn_param_v.get("authenticationSecrets", {}).get(
                "token", None
            )
            self.api_version = conn_param_v.get("serviceProviderProperties", {}).get(
                "api_version", None
            )
            self.service_provider = conn_param_v.get("serviceProvider", None)

            if self.service_provider == "OpenAI":
                self.deployment_name = conn_param_v.get(
                    "serviceProviderProperties", {}
                ).get("model_id", None)
            elif self.service_provider in [
                "AWS",
                "Amazon Bedrock",
                "Bedrock",
                "Others",
            ]:
                spp = conn_param_v.get("serviceProviderProperties", {})
                self.model_id = spp.get("model_id", "us.amazon.nova-premier-v1:0")
                self.region_name = spp.get("region_name", "us-east-1")
            else:
                self.deployment_name = conn_param_v.get(
                    "serviceProviderProperties", {}
                ).get("deployment_name", None)

        else:
            # Legacy/simple connection file without "version" key
            self.service_provider = conn_params.get("service_provider", None)
            if self.service_provider in ["AWS", "Amazon Bedrock", "Bedrock", "Others"]:
                # Expecting keys: model_id, region_name, and api_key
                self.model_id = conn_params.get(
                    "model_id", "us.amazon.nova-premier-v1:0"
                )
                self.region_name = conn_params.get("region_name", "us-east-1")
                self.api_key = conn_params.get("api_key", None)
            else:
                self.azure_endpoint = conn_params.get("azure_endpoint", None)
                self.api_key = conn_params.get("api_key", None)
                self.api_version = conn_params.get("api_version", None)
                self.service_provider = conn_params.get("service_provider", None)
                self.deployment_name = conn_params.get("deployment_name", None)

        if self.service_provider == "AzureOpenAI" or self.service_provider == "Azure":
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )

        elif self.service_provider == "OpenAI":
            self.client = OpenAI(api_key=self.api_key)

        elif self.service_provider == "local-llama":
            self.client = "llama"
            import torch

            self.torch = torch
            from transformers import MllamaForConditionalGeneration, AutoProcessor

            self.model = MllamaForConditionalGeneration.from_pretrained(
                "meta-llama/Llama-3.2-11B-Vision-Instruct",
                torch_dtype=self.torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device_id)
            self.processor = AutoProcessor.from_pretrained(
                "meta-llama/Llama-3.2-11B-Vision-Instruct"
            )
        elif self.service_provider in ["AWS", "Amazon Bedrock", "Bedrock", "Others"]:
            # Use simple HTTP client flow for Bedrock Converse API
            self.client = "aws"
        else:
            # Log additional context for troubleshooting using message logger
            try:
                import traceback

                context = {
                    "service_provider": self.service_provider,
                    "has_api_key": bool(getattr(self, "api_key", None)),
                    "has_deployment_name": bool(getattr(self, "deployment_name", None)),
                    "has_model_id": bool(getattr(self, "model_id", None)),
                    "region_name": getattr(self, "region_name", None),
                }
                _append_message_log("Unknown provider encountered in getConfiguration")
                _append_message_log("Context: " + json.dumps(context))
                _append_message_log(traceback.format_exc())
            except Exception:
                pass
            raise Exception("Unknown provider")

        return {
            "CropSizeFixed": int(self.json_info["CropSizeFixed"]),
            "BlackenAroundFeature": int(self.json_info["BlackenAroundFeature"]),
            "extractBands": tuple(self.json_info["ExtractBands"]),
            "tx": self.json_info["ImageWidth"],
            "ty": self.json_info["ImageHeight"],
            "batch_size": self.batch_size,
            "inheritProperties": 2 | 4 | 8,
            "inputMask": True,
        }

    def getFields(self):

        return json.dumps({"fields": features["fields"]})

    def getGeometryType(self):
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):

        # set pixel values in invalid areas to 0
        rasters_mask = pixelBlocks["rasters_mask"]
        rasters_pixels = pixelBlocks["rasters_pixels"]

        for i in range(0, len(rasters_pixels)):
            rasters_pixels[i][np.where(rasters_mask[i] == 0)] = 0

        pixelBlocks["rasters_pixels"] = rasters_pixels

        polygon_list = []
        scores = []
        labels = []
        reasons = []
        ff = []

        # Prepare and log prompt/context once for the whole run
        base_instruction = (
            "You are an AI Image Classifier that helps people classify images to one of the several provided categories. "
            "Also, share the reasoning behind your classification. You must return the response as a json with the following keys: 1. Classification and 2. Reason. The provided categories are: "
        )
        sys_prompt_once = self.additional_context + base_instruction + str(self.classes)
        try:
            _append_message_log("=== VisionLanguageClassification run start ===")
            _append_message_log("Context: " + str(self.additional_context))
            _append_message_log("Classes: " + str(self.classes))
            _append_message_log(
                "Strict classification: " + str(self.strict_classification)
            )
            _append_message_log("System message: " + sys_prompt_once)
        except Exception:
            pass

        # Pre-flight: for AWS provider, send a simple text-only test message to catch config/auth errors early
        if getattr(self, "client", None) == "aws":
            try:
                test_url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com/model/{self.model_id}/converse"
                test_payload = {
                    "messages": [
                        {"role": "user", "content": [{"text": "Hi How are you"}]}
                    ],
                    "inferenceConfig": {"maxTokens": 64, "temperature": 0.0},
                }
                test_headers = {
                    "Content-Type": "application/json",
                    "Authorization": (
                        f"Bearer {self.api_key}" if self.api_key else None
                    ),
                }
                test_headers = {k: v for k, v in test_headers.items() if v is not None}
                _append_message_log(
                    json.dumps(
                        {
                            "aws_preflight_request": {
                                "url": test_url,
                                "region": self.region_name,
                                "model_id": self.model_id,
                            }
                        }
                    )
                )
                test_resp = requests.post(
                    test_url, json=test_payload, headers=test_headers, timeout=30
                )
                status = test_resp.status_code
                rid = test_resp.headers.get(
                    "x-amzn-requestid"
                ) or test_resp.headers.get("x-amzn-request-id")
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
            except Exception as e:
                try:
                    import traceback

                    _append_message_log(
                        json.dumps(
                            {
                                "aws_preflight_exception": {
                                    "message": str(e),
                                    "trace": traceback.format_exc(),
                                    "region": getattr(self, "region_name", None),
                                    "model_id": getattr(self, "model_id", None),
                                }
                            }
                        )
                    )
                except Exception:
                    pass

        for image in pixelBlocks["rasters_pixels"]:
            _, height, width = image.shape
            temp = np.moveaxis(image, 0, -1)
            pil_image = Image.fromarray(np.moveaxis(image, 0, -1))

            buff = BytesIO()
            pil_image.save(buff, format="JPEG")
            raw_bytes = buff.getvalue()
            encoded_image = base64.b64encode(raw_bytes).decode("utf-8")
            raw_size_bytes = len(raw_bytes)
            encoded_size_chars = len(encoded_image)

            # Use the single, pre-built system prompt for all tiles
            sys_prompt = sys_prompt_once

            if self.client == "llama":
                answer = process_llama_request(self, temp)
            elif self.client == "aws":
                # Amazon Bedrock Converse API request
                try:
                    url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com/model/{self.model_id}/converse"
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
                        "Authorization": (
                            f"Bearer {self.api_key}" if self.api_key else None
                        ),
                    }
                    # Remove None headers
                    headers = {k: v for k, v in headers.items() if v is not None}

                    # Log request metadata (no secrets)
                    try:
                        _append_message_log(
                            json.dumps(
                                {
                                    "aws_request": {
                                        "url": url,
                                        "region": self.region_name,
                                        "model_id": self.model_id,
                                        "image_dims": {
                                            "width": int(width),
                                            "height": int(height),
                                        },
                                        "jpeg_bytes_len": raw_size_bytes,
                                        "base64_len": encoded_size_chars,
                                    }
                                }
                            )
                        )
                    except Exception:
                        pass

                    resp = requests.post(url, json=payload, headers=headers, timeout=60)
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
                        request_id = resp.headers.get(
                            "x-amzn-requestid"
                        ) or resp.headers.get("x-amzn-request-id")
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
                            resp_json.get("output", {})
                            .get("message", {})
                            .get("content", [])
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
                        answer = _extract_json_from_text(output_text)
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
                        answer = {"Classification": "Unknown", "Reason": "Unknown"}
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
                    answer = {"Classification": "Unknown", "Reason": "Unknown"}
                except Exception:
                    import traceback

                    try:
                        error_msg = traceback.format_exc()
                        _append_message_log(
                            json.dumps(
                                {
                                    "aws_request_exception": {
                                        "message": str(error_msg),
                                        "region": getattr(self, "region_name", None),
                                        "model_id": getattr(self, "model_id", None),
                                    }
                                }
                            )
                        )
                    except Exception:
                        # Don't let logging failures affect the main flow
                        pass
                    answer = {"Classification": "Unknown", "Reason": "Unknown"}
            else:
                try:
                    response = self.client.chat.completions.create(
                        model=self.deployment_name,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "\n"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{encoded_image}"
                                        },
                                    },
                                ],
                            },
                        ],
                    )
                    answer = json.loads(response.choices[0].message.content)
                except:
                    answer = {"Classification": "Unknown", "Reason": "Unknown"}

            # Build a closed rectangle ring in row/col (y,x) order, then converted below to (x,y)
            rings = [
                [0, 0],
                [0, max(0, width - 1)],
                [max(0, height - 1), max(0, width - 1)],
                [max(0, height - 1), 0],
                [0, 0],  # close ring
            ]
            polygon_list.append(rings)
            scores.append(100.0)
            # Log the raw classification/reason the model returned (before normalization)
            try:
                # Follow the original structure, but normalize and message/log
                cls_val = (
                    answer.get("Classification") if isinstance(answer, dict) else None
                )
                rsn_val = answer.get("Reason") if isinstance(answer, dict) else None

                try:
                    _append_message_log(
                        "Model Classification (raw): "
                        + (str(cls_val) if cls_val is not None else "None")
                    )
                    _append_message_log(
                        "Model Reason (raw): "
                        + (str(rsn_val) if rsn_val is not None else "None")
                    )
                except Exception:
                    pass

                missing_cls = not isinstance(cls_val, str) or not cls_val.strip()
                missing_rsn = not isinstance(rsn_val, str) or not rsn_val.strip()

                if missing_cls:
                    cls_val = "Unknown"
                if missing_rsn:
                    rsn_val = "Unknown"

                if self.strict_classification:
                    if cls_val in self.classes_list:
                        labels.append(cls_val)
                        reasons.append(rsn_val)
                    else:
                        labels.append("Unknown")
                        reasons.append(rsn_val)
                else:
                    labels.append(cls_val)
                    reasons.append(rsn_val)

                # Log messages to a text file instead of using arcpy.AddMessage
                _append_message_log(f"Classification: {labels[-1]}")
                _append_message_log(f"Reason: {reasons[-1]}")

                # Log raw answer content for diagnostics
                try:
                    _append_message_log("Raw answer: " + json.dumps(answer))
                except Exception:
                    try:
                        _append_message_log("Raw answer (str): " + str(answer))
                    except Exception:
                        pass

                # Log to file if response missing/nulls
                if missing_cls or missing_rsn:
                    try:
                        _append_message_log(
                            "Missing or empty fields from model response"
                        )
                        _append_message_log("Response: " + json.dumps(answer))
                        _append_message_log(f"Used Classification: {labels[-1]}")
                        _append_message_log(f"Used Reason: {reasons[-1]}")
                    except Exception:
                        pass
            except Exception:
                # Handle rare cases where keys are absent or answer is not a dict
                labels.append("Unknown")
                reasons.append("Unknown")
                # Log messages for visibility
                _append_message_log("Classification: Unknown")
                _append_message_log("Reason: Unknown")
                # Log the failure details
                try:
                    import traceback

                    _append_message_log("Exception while reading model response keys")
                    _append_message_log(
                        "Response: "
                        + (
                            json.dumps(answer)
                            if isinstance(answer, dict)
                            else str(answer)
                        )
                    )
                    _append_message_log(traceback.format_exc())
                except Exception:
                    pass

        features["features"] = []
        features["fieldAliases"].update({"Label": "Label"})

        Labelfield = {"name": "Label", "type": "esriFieldTypeString", "alias": "Label"}

        if not Labelfield in features["fields"]:
            features["fields"].append(Labelfield)

        for i in range(len(polygon_list)):
            rings = [[]]
            for j in range(len(polygon_list[i])):
                rings[0].append([polygon_list[i][j][1], polygon_list[i][j][0]])

            features["features"].append(
                {
                    "attributes": {
                        "OID": i + 1,
                        "Confidence": 100.0,
                        "Label": labels[i],
                        "Reason": reasons[i],
                    },
                    "geometry": {"rings": rings},
                }
            )

        return {"output_vectors": json.dumps(features)}
