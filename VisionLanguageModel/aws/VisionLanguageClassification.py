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
        import json, keyring

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
            elif self.service_provider == "AWS":  # NEW AWS support
                spp = conn_param_v.get("serviceProviderProperties", {})
                self.model_id = spp.get("model_id", None)
                if not self.model_id:
                    raise Exception(
                        "AWS model_id is required in connection file serviceProviderProperties"
                    )
                self.region_name = spp.get("aws_region_name", None)
                if not self.region_name:
                    raise Exception(
                        "AWS aws_region_name is required in connection file serviceProviderProperties"
                    )
                # AWS access key ID from serviceProviderProperties
                self.aws_access_key_id = spp.get("aws_access_key", None)
                if not self.aws_access_key_id:
                    raise Exception(
                        "AWS aws_access_key is required in connection file serviceProviderProperties"
                    )
                # AWS secret access key from authenticationSecrets token
                if not self.api_key:
                    raise Exception(
                        "AWS secret access key is required in connection file authenticationSecrets"
                    )
            else:
                self.deployment_name = conn_param_v.get(
                    "serviceProviderProperties", {}
                ).get("deployment_name", None)

        else:
            self.azure_endpoint = conn_params.get("azure_endpoint", None)
            self.api_key = conn_params.get("api_key", None)
            self.api_version = conn_params.get("api_version", None)
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
        elif self.service_provider == "AWS":  # NEW AWS support
            from aws_client import AWSBedrockClient

            self.client = AWSBedrockClient(
                model_id=self.model_id,
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.api_key,
            )
        else:
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

        # final_string = self.prompt

        for image in pixelBlocks["rasters_pixels"]:
            _, height, width = image.shape
            temp = np.moveaxis(image, 0, -1)
            pil_image = Image.fromarray(np.moveaxis(image, 0, -1))

            buff = BytesIO()
            pil_image.save(buff, format="JPEG")
            #
            encoded_image = base64.b64encode(buff.getvalue()).decode("utf-8")

            sys_prompt = "You are an AI Image Classifier that helps people classify images to one of the several provided categories. Also, share the reasoning behind your classification. You must return the response as a json with the following keys: 1. Classification and 2. Reason. The provided categories are: "
            sys_prompt = self.additional_context + sys_prompt + str(self.classes)

            if self.client == "llama":
                answer = process_llama_request(self, temp)
            elif hasattr(self.client, "classify_image"):  # NEW AWS support
                answer = self.client.classify_image(
                    encoded_image, sys_prompt, width, height
                )
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

            rings = [[0, 0], [0, width - 1], [height - 1, width - 1], [height - 1, 0]]
            polygon_list.append(rings)
            scores.append("80")
            try:
                if self.strict_classification:
                    if answer["Classification"] in self.classes_list:
                        labels.append(answer["Classification"])
                        reasons.append(answer["Reason"])
                    else:
                        labels.append("Unknown")
                        reasons.append(answer["Reason"])
                else:
                    labels.append(answer["Classification"])
                    reasons.append(answer["Reason"])
            except:
                labels.append(
                    "Unknown"
                )  # Handle cases where the LLM does not include the key Classification in its output. Happens very rarely.
                reasons.append(answer["Reason"])

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
                        "Confidence": "100",
                        "Label": labels[i],
                        "Reason": reasons[i],
                    },
                    "geometry": {"rings": rings},
                }
            )

        return {"output_vectors": json.dumps(features)}
