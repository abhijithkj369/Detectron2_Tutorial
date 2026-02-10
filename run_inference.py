import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common libraries
import numpy as np
import os, json, cv2, random

# Import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def main():
    # 1. Load the image
    im = cv2.imread(r"D:\AI_PROJECTS\Abhijith\Detectron2\Data\ChatGPT Image Nov 13, 2025, 10_09_22 AM.png")
    if im is None:
        print("Error: 'input.jpg' not found. Please put an image in this folder.")
        return

    # 2. Configure the model
    cfg = get_cfg()
    
    # Use a Model from the Zoo (ResNet-50 Base Model)
    # This automatically downloads the config and the pre-trained weights
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # Use CUDA (GPU) - Crucial check!
    cfg.MODEL.DEVICE = "cuda"
    print(f"Target Device: {cfg.MODEL.DEVICE}")

    # Set confidence threshold (0.5 = 50%)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    
    # 3. Create the Predictor
    print("Building model... (this might take a minute to download weights)")
    predictor = DefaultPredictor(cfg)
    
    # 4. Run Inference
    print("Running inference...")
    outputs = predictor(im)
    
    # 5. Visualize and Save
    print("Saving results...")
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Save output image
    cv2.imwrite("output.jpg", out.get_image()[:, :, ::-1])
    print("Success! Check 'output.jpg' to see the result.")

if __name__ == "__main__":
    main()