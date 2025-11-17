import logging
import os
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image


from fiftyone import Model
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
from fiftyone.core.labels import Detection, Detections, Keypoint, Keypoints, Classification, Classifications
from fiftyone.utils.torch import GetItem
from transformers import AutoModelForCausalLM
from transformers.utils.import_utils import is_flash_attn_2_available

MOONDREAM_OPERATIONS = {
    "caption": {
        "params": {"length": ["short", "normal", "long"]},
    },
    "query": {
        "params": {},
    },
    "detect": {
        "params": {},
    },
    "point": {
        "params": {},
    },
    "classify": {
        "params": {},
    }
}

logger = logging.getLogger(__name__)

# Utility functions
def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Moondream3GetItem(GetItem):
    """GetItem transform for loading images for Moondream3 model.
    
    This class handles data loading in parallel DataLoader workers.
    It should only perform I/O operations (loading images from disk),
    not model inference or heavy preprocessing.
    """
    
    @property
    def required_keys(self):
        """Keys needed from each sample."""
        return ["filepath"]
    
    def __call__(self, sample_dict):
        """Load and return a PIL Image.
        
        This runs in DataLoader worker processes for parallel loading.
        
        Args:
            sample_dict: Dict with keys from required_keys
            
        Returns:
            PIL.Image: RGB image
        """
        filepath = sample_dict["filepath"]
        image = Image.open(filepath).convert("RGB")
        return image


class Moondream3(Model, SupportsGetItem, TorchModelMixin):
    """A FiftyOne model for running the Moondream2 model on images.
    
    Args:
        model_path (str): Path to model or HuggingFace model name
        operation (str, optional): Type of operation to perform
        prompt (str, optional): Prompt text to use
        **kwargs: Additional parameters
    """

    def __init__(
        self, 
        model_path: str,
        operation: str = None,
        prompt: str = None,
        **kwargs
    ):
        # Initialize base classes
        SupportsGetItem.__init__(self)
        
        # REQUIRED: Set preprocessing flag
        # GetItem handles data loading, so preprocessing happens there
        self._preprocess = False
        
        if not model_path:
            raise ValueError("model_path is required")
            
        self.model_path = model_path
        self._operation = None
        self._prompt = prompt
        self.params = {}
        self._fields = {}
        
        
        # Set operation if provided
        if operation:
            self.operation = operation
            
        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
               
        # Set device
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading model from local path: {model_path}")

        print("\n" + "="*80)
        print("NOTICE: Creating necessary symbolic links for custom model code")
        print("When loading Moondream3 from a local directory,")
        print("the Transformers library expects to find Python modules in:")
        print(f"  ~/.cache/huggingface/modules/transformers_modules/moondream3/")
        print("rather than in your downloaded model directory.")
        print("Creating symbolic links to connect these locations...")
        print("="*80 + "\n")

        cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/moondream3")

        os.makedirs(cache_dir, exist_ok=True)
        # Find all Python files in the model directory and create symlinks
        for file in os.listdir(model_path):
            if file.endswith('.py'):
                src = os.path.join(model_path, file)
                dst = os.path.join(cache_dir, file)
                # Create a symlink instead of copying
                if not os.path.exists(dst):
                    print(f"Creating symlink for {file}")
                    os.symlink(src, dst)

            model_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
            }

        # Try bfloat16 on capable CUDA devices, otherwise use float16
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            
            # Use bfloat16 on Ampere+ GPUs (SM 8.0+), float16 on older GPUs
            if capability[0] >= 8:
                model_kwargs["torch_dtype"] = torch.bfloat16
                logger.info("Using bfloat16 on Ampere+ GPU")
            else:
                model_kwargs["torch_dtype"] = torch.float16
                logger.info("Using float16 on pre-Ampere GPU")
                
            model_kwargs["device_map"] = self.device
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            **model_kwargs
        )

        self.model.compile()
        self.model.eval()

    @property
    def media_type(self):
        return "image"
    
    # ============ REQUIRED PROPERTIES FROM Model BASE CLASS ============
    
    @property
    def transforms(self):
        """Preprocessing transforms applied to inputs.
        
        For SupportsGetItem models, preprocessing happens in GetItem,
        so return None.
        """
        return None
    
    @property
    def preprocess(self):
        """Whether model should apply preprocessing.
        
        Returns False because GetItem handles data loading and preprocessing.
        """
        return self._preprocess
    
    @preprocess.setter
    def preprocess(self, value):
        """Allow FiftyOne to control preprocessing."""
        self._preprocess = value
    
    @property
    def ragged_batches(self):
        """Whether this model supports batches with varying sizes.
        
        MUST return False to enable batching! Even though our inputs
        (PIL Images) have variable sizes, we return False because:
        1. We provide a custom collate_fn that handles variable sizes
        2. ragged_batches=True would disable batching entirely
        3. Our model handles variable-size inputs internally
        """
        return False
    
    # ============ REQUIRED PROPERTIES FROM TorchModelMixin ============
    
    @property
    def has_collate_fn(self):
        """Whether this model provides custom batch collation.
        
        Return True because we need custom batching logic for
        variable-size images (don't stack them into tensors).
        """
        return True
    
    @property
    def collate_fn(self):
        """Custom collation function for batching.
        
        Returns a function that keeps images as a list rather than
        trying to stack them (which would fail for variable sizes).
        """
        @staticmethod
        def identity_collate(batch):
            """Return batch as-is without stacking.
            
            Args:
                batch: List of PIL Images from GetItem
                
            Returns:
                List of PIL Images (unchanged)
            """
            return batch
        return identity_collate
    
    # ============ MODEL-SPECIFIC PROPERTIES ============
    
    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields

    def _get_field(self):
        """Get the field name to use for prompt extraction."""
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)
        return prompt_field

    @property
    def operation(self):
        """Get the current operation."""
        return self._operation

    @operation.setter
    def operation(self, value):
        """Set the operation with validation."""
        if value not in MOONDREAM_OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(MOONDREAM_OPERATIONS.keys())}")
        self._operation = value

    @property
    def prompt(self):
        """Get the current prompt text."""
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        """Set the prompt text."""
        self._prompt = value
        
    @property
    def length(self):
        """Get the caption length."""
        return self.params.get("length", "normal")

    @length.setter
    def length(self, value):
        """Set the caption length with validation."""
        valid_lengths = MOONDREAM_OPERATIONS["caption"]["params"]["length"]
        if value not in valid_lengths:
            raise ValueError(f"Invalid length: {value}. Must be one of {valid_lengths}")
        self.params["length"] = value

    # ============ REQUIRED METHODS FROM SupportsGetItem ============
    
    def build_get_item(self, field_mapping=None):
        """Build the GetItem transform for data loading.
        
        This is called by FiftyOne to create the data loading pipeline.
        
        Args:
            field_mapping: Optional dict mapping required_keys to dataset fields
                          e.g., {"filepath": "image_path"}
        
        Returns:
            Moondream3GetItem instance for loading images
        """
        return Moondream3GetItem(field_mapping=field_mapping)

    # ============ HELPER METHODS ============

    def _convert_to_detections(self, boxes: List[Dict[str, float]], label: str) -> Detections:
        """Convert Moondream2 detection output to FiftyOne Detections.
        
        Args:
            boxes: List of bounding box dictionaries
            label: Object type label
            
        Returns:
            FiftyOne Detections object
        """
        detections = []

        for box in boxes:
            detection = Detection(
                label=label,
                bounding_box=[
                    box["x_min"],
                    box["y_min"],
                    box["x_max"] - box["x_min"],  # width
                    box["y_max"] - box["y_min"]   # height
                ]
            )

            detections.append(detection)
        
        return Detections(detections=detections)

    def _convert_to_keypoints(self, points: List[Dict[str, float]], label: str) -> Keypoints:
        """Convert Moondream2 point output to FiftyOne Keypoints.
        
        Args:
            points: List of point dictionaries
            label: Object type label
            
        Returns:
            FiftyOne Keypoints object
        """
        keypoints = []

        for idx, point in enumerate(points):

            keypoint = Keypoint(
                label=f"{label}",
                points=[[point["x"], point["y"]]]
            )

            keypoints.append(keypoint)
        
        return Keypoints(keypoints=keypoints)

    def _predict_caption(self, image: Image.Image, sample=None) -> Dict[str, str]:
        """Generate a caption for an image.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Caption result
        """
        length = self.params.get("length", "normal")
        with torch.no_grad():
            result = self.model.caption(image, length=length)["caption"]

        return result.strip()

    def _predict_query(self, image: Image.Image, sample=None) -> Dict[str, str]:
        """Answer a visual query about an image.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Query answer
        """
        if not self.prompt:
            raise ValueError("No prompt provided for query operation")

        with torch.no_grad():    
            result = self.model.query(image, self.prompt)["answer"]

        return result.strip()

    def _predict_detect(self, image: Image.Image, sample=None) -> Detections:
        """Detect objects in an image with multiple class support.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            FiftyOne Detections object containing all detected objects
        """
        if not self.prompt:
            raise ValueError("No prompt provided for detect operation")
        
        # Handle different types of prompt inputs (list or string)
        if isinstance(self.prompt, list):
            # If prompt is already a list, use it directly
            classes_to_find = self.prompt
            logger.info(f"Using list of classes: {classes_to_find}")
        else:
            # If prompt is a string, split it by commas and clean up whitespace
            classes_to_find = [cls.strip() for cls in self.prompt.split(',')]
            logger.info(f"Parsed classes from string: {classes_to_find}")
        
        # Initialize an empty list to store all detections across all classes
        all_detections = []
        
        # Process each class separately since Moondream2 only supports one class at a time
        for cls in classes_to_find:
            logger.info(f"Detecting class: {cls}")
            
            # Call the Moondream2 model to detect objects of this specific class
            with torch.no_grad():
                result = self.model.detect(image, cls)["objects"]
            logger.info(f"Found {len(result)} instances of '{cls}'")
            
            # Convert the detected objects to FiftyOne Detection format using our helper method
            # This returns a Detections object for the current class
            class_detections = self._convert_to_detections(result, cls)
            
            # Add all detections from this class to our combined list
            all_detections.extend(class_detections.detections)
        
        # Return all detections combined in a single FiftyOne Detections object
        logger.info(f"Total objects detected across all classes: {len(all_detections)}")
        return Detections(detections=all_detections)

    def _predict_point(self, image: Image.Image, sample=None) -> Keypoints:
        """Identify point locations of objects in an image with multiple class support.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            FiftyOne Keypoints object containing all detected keypoints
        """
        if not self.prompt:
            raise ValueError("No prompt provided for point operation")
        
        # Handle different types of prompt inputs (list or string)
        if isinstance(self.prompt, list):
            # If prompt is already a list, use it directly
            classes_to_find = self.prompt
            logger.info(f"Using list of classes: {classes_to_find}")
        else:
            # If prompt is a string, split it by commas and clean up whitespace
            classes_to_find = [cls.strip() for cls in self.prompt.split(',')]
            logger.info(f"Parsed classes from string: {classes_to_find}")
        
        # Initialize an empty list to store all keypoints across all classes
        all_keypoints = []
        
        # Process each class separately since Moondream2 only supports one class at a time
        for cls in classes_to_find:
            logger.info(f"Finding points for class: {cls}")
            
            # Call the Moondream2 model to find points for this specific class
            with torch.no_grad():
                result = self.model.point(image, cls)["points"]
            logger.info(f"Found {len(result)} points for '{cls}'")
            
            # Convert the detected points to FiftyOne Keypoint format using our helper method
            # This returns a Keypoints object for the current class
            class_keypoints = self._convert_to_keypoints(result, cls)
            
            # Add all keypoints from this class to our combined list
            all_keypoints.extend(class_keypoints.keypoints)
        
        # Return all keypoints combined in a single FiftyOne Keypoints object
        logger.info(f"Total points detected across all classes: {len(all_keypoints)}")
        return Keypoints(keypoints=all_keypoints)
        
    def _to_classifications(self, classes: str) -> Classifications:
        """Convert a list of classification dictionaries to FiftyOne Classifications.
        
        Args:
            classes: List of dictionaries containing classification information.
                Each dictionary should have:
                - 'label': String class label
                
        Returns:
            fo.Classifications object containing the converted classification annotations
        """
        classification = Classification(
                    label=classes,
                )
        return Classifications(classifications=[classification])
    

    def _predict_classify(self, image: Image.Image, sample=None) -> Classifications:
        """Classify an image based on the prompt.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            fo.Classifications: Classification results
        """
        # Check if a classification prompt was provided
        if not self.prompt:
            raise ValueError("No prompt provided for classify operation")
        
        # Handle different types of prompt inputs (list or string)
        if isinstance(self.prompt, list):
            # If prompt is a list of classes, format it as a special classification prompt
            classes_list = ", ".join(self.prompt)  # Join classes with commas for display
            classification_prompt = f"A photo of a: {classes_list}. Respond with only of the provided choices name, and nothing more."
            logger.info(f"Using list of classes for classification: {self.prompt}")
        else:
            # If prompt is a string, use it directly
            classification_prompt = f"A photo of a: {self.prompt}. Respond with only of the provided choices name, and nothing more."
            logger.info(f"Using string prompt for classification: {self.prompt}")
        
        # Query the model with the image and prompt, extract and clean the response
        logger.info(f"Sending classification prompt to model")
        with torch.no_grad():
            result = self.model.query(image, classification_prompt)["answer"].strip()
        logger.info(f"Model classified image as: {result}")
        
        # Convert the string result to FiftyOne Classifications format
        classifications = self._to_classifications(result)
        
        return classifications

    def _predict(self, image: Image.Image, sample=None) -> Any:
        """Process a single image with Moondream2.
        
        Args:
            image: PIL image
            sample: Optional FiftyOne sample
            
        Returns:
            Operation results (various types depending on operation)
        """
        # Centralized field handling
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                self._prompt = str(field_value)
                
        if not self.operation:
            raise ValueError("No operation has been set")
                
        prediction_methods = {
            "caption": self._predict_caption,
            "query": self._predict_query,
            "detect": self._predict_detect,
            "point": self._predict_point,
            "classify": self._predict_classify
        }
        
        predict_method = prediction_methods.get(self.operation)

        if predict_method is None:
            raise ValueError(f"Unknown operation: {self.operation}")
            
        return predict_method(image, sample)

    def predict(self, image: np.ndarray, sample=None) -> Dict[str, Any]:
        """Process an image array with Moondream2.
        
        Args:
            image: numpy array image
            sample: Optional FiftyOne sample
            
        Returns:
            dict: Operation results
        """
        logger.info(f"Running {self.operation} operation")
        pil_image = Image.fromarray(image)
        return self._predict(pil_image, sample)
    
    def predict_all(self, images, preprocess=None):
        """Process a batch of images with Moondream3.
        
        This method enables efficient batch processing when using
        dataset.apply_model() with batch_size > 1.
        
        Args:
            images: List of PIL Images from GetItem transform
            preprocess: Whether to apply preprocessing (uses self._preprocess if None)
                       Not used since GetItem handles data loading
        
        Returns:
            List of predictions (same length as images)
            Each prediction is in FiftyOne format:
            - Detections for "detect" operation
            - Keypoints for "point" operation  
            - Classifications for "classify" operation
            - str for "caption" and "query" operations
        """
        if preprocess is None:
            preprocess = self._preprocess
        
        if not self.operation:
            raise ValueError("No operation has been set")
        
        logger.info(f"Running batch {self.operation} operation on {len(images)} images")
        
        # Process each image in the batch
        # Note: Moondream3 model may not have native batch methods for all operations,
        # so we process images individually. However, we still benefit from:
        # 1. Parallel data loading via DataLoader workers
        # 2. Better GPU utilization through batching
        # 3. Amortized overhead
        results = []
        for image in images:
            # images are already PIL Images from GetItem
            result = self._predict(image, sample=None)
            results.append(result)
        
        return results