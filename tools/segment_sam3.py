from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor 
from utils.config import SAM3_PATH

 
class SegementModule:
    def __init__(
            self, 
            device="cuda",
        ):
        self.device = device
        self.model = build_sam3_image_model(check_point_path=SAM3_PATH, device=device)
        self.processor = Sam3Processor(self.model, device=device)
        
    
    def run_segment(
        self, 
        image_path: str,     
        prompt: str  
    ):
        image = Image.open(image_path)
        inference_state = self.processor.set_image(image)
        # Prompt the model with text
        output = self.processor.set_text_prompt(state=inference_state, 
                                                prompt=prompt)

        # Get the masks, bounding boxes, and scores
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

        return masks, boxes, scores
    

