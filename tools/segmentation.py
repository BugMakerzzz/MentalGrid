from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import numpy as np
from PIL import Image
import torch
 
class SegmentationModule:
    def __init__(
            self,
            config, 
            device="cuda",
        ):
        self.config = config
        self.device = device
        self.sam_model = build_sam2(config.config, config.ckpt_path, device=device)
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)

        self.processor = AutoProcessor.from_pretrained(config.grounding_path)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            config.grounding_path
        ).to(device)
        

    def make_text_prompt(self, prompts):
        """
        GroundingDINO HF demo requirement:
        - lowercased
        - each query ends with a dot
        e.g. ["car", "tire"] -> "car. tire."
        """
        prompts = [p.strip().lower().rstrip(".") for p in prompts]
        return ". ".join(prompts) + "."


    def run_segmentation(self, image_path, prompt):
        """
        EXACTLY follow your HF demo style:
        - processor(images=image, text=text)
        - grounding_model(**inputs)
        - processor.post_process_grounded_object_detection(...)
        - input_boxes -> sam2_predictor.predict(box=...)
        Return:
        masks: (N, H, W) bool
        boxes: (N, 4) float xyxy
        labels: list[str]
        scores: list[float]
        """
        
        image = Image.open(image_path).convert("RGB")
        
        self.sam_predictor.set_image(np.array(image))
        
        if isinstance(prompt, list):
            prompt = self.make_text_prompt(prompts=prompt)

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[image.size[::-1]]  # (H, W)
        )

        boxes = results[0]["boxes"].cpu().numpy()     # (N, 4)
        labels = results[0]["labels"]                # list[str]
        scores = results[0]["scores"].cpu().numpy().tolist()

        if len(boxes) == 0:
            return None

        masks, sam_scores, logits = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        masks = masks.astype(bool)  # (N, H, W)
        
        return masks, boxes, labels, scores