import torch
from moge.model.v2 import MoGeModel 
import cv2

class DepthModule:
    def __init__(self, config, device) -> None:
        self.moge2_model = MoGeModel.from_pretrained(MOGE2_MODEL_ID).to(device)
        self.device = device
    
    def run_moge(self, image_bgr):
        """
        Follow your MoGe-2 demo:
        input_image: (3,H,W) float32 RGB [0,1]
        output keys: points, depth, mask, intrinsics, normal(optional)
        """
        input_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = torch.tensor(input_image / 255.0, dtype=torch.float32, device=self.device).permute(2,0,1)
        output = self.moge_model.infer(input_tensor)
        points = output["points"].detach().cpu().numpy()   # (H, W, 3)
        valid = output["mask"].detach().cpu().numpy().astype(bool)  # (H, W)
        return points, valid, output

    def extract_object_points(self, points, valid_mask, obj_mask):
        sel = valid_mask & obj_mask
        pts = points[sel]
        return pts

    def centroid(self, pts):
        if pts.shape[0] == 0:
            return None
        return pts.mean(axis=0)


    def run_depth(self, image_path, seg_out):
        image = cv2.imread(image_path)
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if seg_out is None:
                return
            masks, boxes, labels, scores = seg_out
            N = masks.shape[0]

            # run MoGe-2 once for the whole scene
            points, valid_mask, moge_out = self.run_moge(image_bgr=image)
        
            # per-instance 3D position
            instances = []
            for i in range(N):
                m = masks[i]
                pts_i = self.extract_object_points(points, valid_mask, m)
                c_i = self.centroid(pts_i)
                if not c_i:
                    continue
                instances.append({
                    "instance_id": i,
                    "label": labels[i],
                    "score": scores[i],
                    "centroid_xyz": c_i.tolist()
                })

            highest_score_by_label = {}
            unique_instances = []
            for inst in instances:
                lab = inst["label"]
                score = inst["score"]
                if lab not in highest_score_by_label:
                    highest_score_by_label[lab] = inst
                else:
                    current_highest_score = highest_score_by_label[lab]["score"]
                    if score > current_highest_score:
                        highest_score_by_label[lab] = inst

            for lab in highest_score_by_label:
                unique_instances.append(highest_score_by_label[lab])
            instances = unique_instances
            # save json
            
            object_locations = {}
            for ins in instances:
                object_locations[ins['label']] = [ins['centroid_xyz'][0], ins['centroid_xyz'][2]]
            
            return object_locations