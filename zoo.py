
import os
import logging
import json
from PIL import Image
from typing import Dict, Any, List, Union, Optional

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from transformers.utils.import_utils import is_flash_attn_2_available

from .modular_isaac import IsaacProcessor

logger = logging.getLogger(__name__)

DEFAULT_DETECTION_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in detecting and localizating meaningful visual elements. 

You can detect and localize objects, components, people, places, things, and UI elements in images using 2D bound boxes.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "detections": [
        {
            "bbox_2d": [x1, y1, x2, y2],
            "label": "descriptive label for the bounding box"
        }
    ]
}
```

The JSON should contain bounding boxes in pixel coordinates [x1,y1,x2,y2] format, where:
- x1,y1 is the top-left corner
- x2,y2 is the bottom-right corner
- Provide specific, descriptive labels for each detected element
- Include all relevant elements that match the user's request
- For UI elements, include their function when possible (e.g., "Login Button" rather than just "Button")
- If many similar elements exist, prioritize the most prominent or relevant ones

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions and detect.

<hint>BOX</hint>
"""


DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """You are a helpful assistant. You specializes in comprehensive classification across any visual domain, capable of analyzing:

Unless specifically requested for single-class output, multiple relevant classifications can be provided.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "classifications": [
        {
            "label": "descriptive class label",
        }
    ]
}
```

The JSON should contain a list of classifications where:
- Each classification must have a 'label' field
- Labels should be descriptive strings describing what you've identified in the image, but limited to one or two word responses
- The response should be a list of classifications
"""

DEFAULT_KEYPOINT_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in key point detection across any visual domain. A key point represents the center of any meaningful visual element. 

Key points should adapt to the context (physical world, digital interfaces, UI elements, etc.) while maintaining consistent accuracy and relevance. 

For each key point identify the key point and provide a contextually appropriate label and always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "keypoints": [
        {
            "point_2d": [x, y],
            "label": "descriptive label for the point"
        }
    ]
}
```

The JSON should contain points in pixel coordinates [x,y] format, where:
- x is the horizontal center coordinate of the visual element
- y is the vertical center coordinate of the visual element
- Include all relevant elements that match the user's request
- You can point to multiple visual elements

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions and point.

<hint>POINT</hint>
"""

DEFAULT_OCR_DETECTION_SYSTEM_PROMPT = """You are a helpful assistant specializing in text detection and recognition (OCR) in images. Your can read, detect, and locate text from any visual content, including documents, UI elements, signs, or any other text-containing regions.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "text_detections": [
        {
            "bbox_2d": [x1, y1, x2, y2],
            "text": "Exact text content found in this region"  // Transcribe text exactly as it appears
        }
    ]
}
```

The JSON should contain bounding boxes in pixel coordinates [x1,y1,x2,y2] format, where:
- x1,y1 is the top-left corner
- x2,y2 is the bottom-right corner
- The 'text' field should be a string containing the exact text content found in the region

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's perform the OCR detections.

<hint>BOX</hint>
"""

DEFAULT_OCR_SYSTEM_PROMPT = """You are an OCR assistant. Your task is to identify and extract all visible text from the image provided. Preserve the original formatting as closely as possible, including:

- Line breaks and paragraphs  
- Headings and subheadings  
- Any tables, lists, bullet points, or numbered items  
- Special characters, spacing, and alignment  

Output strictly the extracted text in Markdown format, reflecting the layout and structure of the original image. Do not add commentary, interpretation, or summarization—only return the raw text content with its formatting.

Respond with 'No Text' if there is no text in the provided image.
"""

DEFAULT_VQA_SYSTEM_PROMPT = "You are a helpful assistant. You provide clear and concise answerss to questions about images. Report answers in natural language text in English."

OPERATIONS = {
    "detect": DEFAULT_DETECTION_SYSTEM_PROMPT,
    "point": DEFAULT_KEYPOINT_SYSTEM_PROMPT,
    "ocr_detection": DEFAULT_OCR_DETECTION_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    "ocr": DEFAULT_OCR_SYSTEM_PROMPT,
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class IsaacModel(SamplesMixin, Model):
    """A FiftyOne model for running Isaac 0.1 vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self._operation = operation
        self.prompt = prompt
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        model_kwargs = {
            "device_map": self.device,
        }

        # Set optimizations based on CUDA device capabilities
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self.device)
            
            # Enable flash attention if available, otherwise use sdpa
            model_kwargs["attn_implementation"] = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
            
            # Enable bfloat16 on Ampere+ GPUs (compute capability 8.0+), otherwise use float16
            model_kwargs["torch_dtype"] = torch.bfloat16 if capability[0] >= 8 else torch.float16

        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
        )
        
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=False
            )
        
        self.config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
            )

        logger.info("Loading processor")

        self.processor = IsaacProcessor.from_pretrained(
            tokenizer=self.tokenizer,
            config=self.config
        )

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else OPERATIONS[self.operation]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    def _parse_json(self, s: str) -> Optional[Dict]:
        """Parse JSON from model output.
        
        The model may return JSON in different formats:
        1. Raw JSON string
        2. JSON wrapped in markdown code blocks (```json ... ```)
        3. Non-JSON string (returns None)
        
        Args:
            s: String output from the model to parse
            
        Returns:
            Dict: Parsed JSON dictionary if successful
            None: If parsing fails or input is invalid
            Original input: If input is not a string
        """
        # Return input directly if not a string
        if not isinstance(s, str):
            return s
            
        # Handle JSON wrapped in markdown code blocks
        if "```json" in s:
            try:
                # Extract JSON between ```json and ``` markers
                s = s.split("```json")[1].split("```")[0].strip()
            except:
                pass
        
        # Attempt to parse the JSON string
        try:
            return json.loads(s)
        except:
            # Log first 200 chars of failed parse for debugging
            logger.debug(f"Failed to parse JSON: {s[:200]}")
            return None

    def _to_detections(self, boxes: List[Dict], image_width: int, image_height: int) -> fo.Detections:
        """Convert bounding boxes to FiftyOne Detections.
        
        Args:
            boxes: List of dictionaries containing bounding box info.
                Each box should have:
                - 'bbox_2d' or 'bbox': List of [x1,y1,x2,y2] coordinates in 0-1000 range from model
                - 'label': Optional string label (defaults to "object")
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels

        Returns:
            fo.Detections object containing the converted bounding box annotations
            with coordinates normalized to [0,1] x [0,1] range
        """
        if not boxes:
            return fo.Detections(detections=[])
            
        detections = []
        
        for box in boxes:
            try:
                # Try to get bbox from either bbox_2d or bbox field
                bbox = box.get('bbox_2d', box.get('bbox', None))
                if not bbox:
                    continue
                    
                # Model outputs coordinates in 0-1000 range, normalize to 0-1
                x1, y1, x2, y2 = map(float, bbox)
                x = x1 / 1000.0  # Normalized left x
                y = y1 / 1000.0  # Normalized top y
                w = (x2 - x1) / 1000.0  # Normalized width
                h = (y2 - y1) / 1000.0  # Normalized height
                
                # Create Detection object with normalized coordinates
                try:
                    detection = fo.Detection(
                        label=str(box.get("label", "object")),
                        bounding_box=[x, y, w, h],
                    )
                    detections.append(detection)
                except:
                    continue
                
            except Exception as e:
                # Log any errors processing individual boxes but continue
                logger.debug(f"Error processing box {box}: {e}")
                continue
                
        return fo.Detections(detections=detections)

    def _to_ocr_detections(
        self, 
        boxes: List[Dict], 
        image_width: int, 
        image_height: int
    ) -> fo.Detections:
        """Convert OCR results to FiftyOne Detections with text content.
        
        Args:
            boxes: List of OCR dictionaries
            image_width: Width of the original image in pixels
            image_height: Height of the original image in pixels
            
        Returns:
            fo.Detections object containing the OCR annotations with text content
        """
        if not boxes:
            return fo.Detections(detections=[])
            
        detections = []
        
        # Process each OCR box
        for box in boxes:
            try:
                # Extract the bounding box coordinates and text content
                bbox = box.get('bbox_2d', box.get('bbox', None))
                text = box.get('text')  # The actual text string
                
                # Skip if missing required bbox or text fields
                if not bbox or not text:
                    continue
                
                # Model outputs coordinates in 0-1000 range, normalize to 0-1
                x1, y1, x2, y2 = map(float, bbox)
                
                # Convert to FiftyOne format: [x, y, width, height]
                x = x1 / 1000.0  # Left coordinate (normalized)
                y = y1 / 1000.0  # Top coordinate (normalized)
                w = (x2 - x1) / 1000.0  # Width (normalized)
                h = (y2 - y1) / 1000.0  # Height (normalized)
                
                # Create Detection object with normalized coordinates
                detection = fo.Detection(
                    label="text",
                    bounding_box=[x, y, w, h],
                    text=str(text),
                )
                detections.append(detection)
                
            except Exception as e:
                # Log any errors processing individual boxes but continue
                logger.debug(f"Error processing OCR box {box}: {e}")
                continue
                
        # Return all detections wrapped in a FiftyOne Detections container
        return fo.Detections(detections=detections)

    def _to_keypoints(self, points: List[Dict], image_width: int, image_height: int) -> fo.Keypoints:
        """Convert a list of point dictionaries to FiftyOne Keypoints.
        
        Args:
            points: List of dictionaries containing point information.
                Each point should have:
                - 'point_2d': List of [x,y] coordinates in 0-1000 range from model
                - 'label': String label describing the point
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
                
        Returns:
            fo.Keypoints object containing the converted keypoint annotations
            with coordinates normalized to [0,1] x [0,1] range
        """
        if not points:
            return fo.Keypoints(keypoints=[])
            
        keypoints = []
        
        for point in points:
            try:
                # Get coordinates from point_2d field and convert to float
                x, y = point["point_2d"]
                x = float(x.cpu() if torch.is_tensor(x) else x)
                y = float(y.cpu() if torch.is_tensor(y) else y)
                
                # Model outputs coordinates in 0-1000 range, normalize to 0-1
                normalized_point = [
                    x / 1000.0,
                    y / 1000.0
                ]
                
                keypoint = fo.Keypoint(
                    label=str(point.get("label", "point")),
                    points=[normalized_point],
                )
                keypoints.append(keypoint)
            except Exception as e:
                logger.debug(f"Error processing point {point}: {e}")
                continue
                
        return fo.Keypoints(keypoints=keypoints)

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert a list of classification dictionaries to FiftyOne Classifications.
        
        Args:
            classes: List of dictionaries containing classification information.
                Each dictionary should have:
                - 'label': String class label
                
        Returns:
            fo.Classifications object containing the converted classification 
            annotations with labels
        """
        if not classes:
            return fo.Classifications(classifications=[])
            
        classifications = []
        
        # Process each classification dictionary
        for cls in classes:
            try:
                # Create Classification object with required label and optional confidence
                classification = fo.Classification(
                    label=str(cls["label"]),  # Convert label to string for consistency
                )
                classifications.append(classification)
            except Exception as e:
                # Log any errors but continue processing remaining classifications
                logger.debug(f"Error processing classification {cls}: {e}")
                continue
                
        # Return Classifications container with all processed results
        return fo.Classifications(classifications=classifications)

    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, fo.Keypoints, fo.Classifications, str]:
        """Process a single image through the model and return predictions.
        
        This internal method handles the core prediction logic including:
        - Constructing the chat messages with system prompt and user query
        - Processing the image and text through the model
        - Parsing the output based on the operation type (detection/points/classification/VQA)
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            One of:
            - fo.Detections: For object detection results
            - fo.Keypoints: For keypoint detection results  
            - fo.Classifications: For classification results
            - str: For VQA text responses
            
        Raises:
            ValueError: If no prompt has been set
        """
        # Use local prompt variable instead of modifying self.prompt
        prompt = self.prompt  # Start with instance default
        
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)  # Local variable, doesn't affect instance

        # Prepare input
        messages = [
            {"role": "system", "type": "text", "content": self.system_prompt},
            {"role": "user", "type": "image", "content": "<image>"},
            {"role": "user", "type": "text", "content": prompt}
        ]
        images = image  # Replace with your image path

        # Process input
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
            )

        inputs = self.processor(
            text=text, 
            images=images, 
            return_tensors="pt"
            )
        
        tensor_stream = inputs["tensor_stream"].to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                tensor_stream=tensor_stream,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode and print output
        output_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if self.operation == "vqa":
            return output_text.strip()
        
        elif self.operation == "ocr":
            return output_text.strip()
        
        elif self.operation == "detect":
            parsed = self._parse_json(output_text) or {}
            data = parsed.get('detections', [])
            input_width, input_height = image.size
            return self._to_detections(data, input_width, input_height)
        
        elif self.operation == "point":
            parsed = self._parse_json(output_text) or {}
            data = parsed.get('keypoints', [])
            input_width, input_height = image.size
            return self._to_keypoints(data, input_width, input_height)
        
        elif self.operation == "classify":
            parsed = self._parse_json(output_text) or {}
            data = parsed.get('classifications', [])
            return self._to_classifications(data)
        
        elif self.operation == "ocr_detection":
            parsed = self._parse_json(output_text) or {}
            data = parsed.get('text_detections', [])
            input_width, input_height = image.size
            return self._to_ocr_detections(data, input_width, input_height)
        
        else:
            return None

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)