import os
import json
import torch
import cv2 as cv
import numpy as np
from typing import List

from nets.nn import Craft
from utils import util


class TextDetector:
    def __init__(
        self,
        detection_weights: str,
        refine_weights,
        device: str = "cuda",
    ):
        self.device = device
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available")
            is_cuda = True
        else:
            is_cuda = False

        self.craft = Craft(
            text_threshold=0.7,
            link_threshold=0.4,
            refiner=True,
            crop_type="box",
            weight_path_craft_net=detection_weights,
            weight_path_refine_net=refine_weights,
            cuda=is_cuda,
        )

    def merge_boxes(self, bbox: List, line_overlap_threshold=0.6) -> List:
        bbox = sorted(bbox, key=lambda x: x[1])
        merged_bbox = []
        for i in range(len(bbox)):
            x_min, y_min, x_max, y_max = bbox[i]
            if i == 0:
                merged_bbox.append([x_min, y_min, x_max, y_max])
            else:
                x_min_prev, y_min_prev, x_max_prev, y_max_prev = merged_bbox[-1]
                y_intersect = (min(y_max, y_max_prev) - max(y_min, y_min_prev)) / (
                    max(y_max, y_max_prev) - min(y_min, y_min_prev)
                )
                if y_intersect > line_overlap_threshold:
                    new_x_min = min(x_min, x_min_prev)
                    new_y_min = min(y_min, y_min_prev)
                    new_x_max = max(x_max, x_max_prev)
                    new_y_max = max(y_max, y_max_prev)
                    merged_bbox[-1] = [new_x_min, new_y_min, new_x_max, new_y_max]
                else:
                    merged_bbox.append([x_min, y_min, x_max, y_max])
        return merged_bbox

    def results_to_json(self, results: List) -> List[dict]:
        json_output = []
        for result in results:
            json_output.append({"bbox": result})
        return json_output

    def detect(self, img_list: List, pad=0.1) -> List:
        print("Text detection")
        os.makedirs("./output/", exist_ok=True)
        
        detection_results = []
        for idx, img in enumerate(img_list):
            print(img.shape)
            img_copy = img.copy()
            results = self.craft.detect_text(img)
            result = np.array(results["boxes"])
            result = result.astype(int)
            bbox = []
            
            for res in result:
                x_min, y_min = res[0, :].tolist()
                x_max, y_max = res[2, :].tolist()
                h = y_max - y_min
                pad_size = int(h * pad)
                x_min = max(0, x_min - pad_size)
                y_min = max(0, y_min - pad_size)
                x_max = min(img.shape[1], x_max + pad_size)
                y_max = min(img.shape[0], y_max + pad_size)

                bbox.append([x_min, y_min, x_max, y_max])
                
                # Draw bounding boxes
                cv.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            bbox = self.merge_boxes(bbox)
            bbox_json = self.results_to_json(bbox)
            detection_results.append(bbox_json)
            
            # Save image result
            cv.imwrite(f"./output/detection_{idx:03d}.jpg", img_copy)
            with open(f"./output/detection_{idx:03d}.json", 'w') as f:
                json.dump(bbox_json, f, indent=2)
        
        return detection_results


def main():
    craft_weight = "./weights/craft_mlt_25k.pth"
    refine_weight = "./weights/craft_refiner_CTW1500.pth"

    img_list = util.read_media("./assets/testimage.png")
    detector = TextDetector(craft_weight, refine_weight, device='cpu')
    detection_results = detector.detect(img_list)
    print(detection_results)


if __name__ == "__main__":
    main()


