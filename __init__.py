import cv2
import numpy as np
from PIL import Image
import torch

class FindBrightestSpot:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("x", "y")
    FUNCTION = "find_spot"
    CATEGORY = "image/opencv"

    def find_spot(self, image):
        print("ComfyUI Image Shape:", image.shape)
        print("ComfyUI Image Tensor:", image)

        img_np = image.cpu().numpy()
        print("NumPy Array Shape:", img_np.shape)
        print("NumPy Array Data (first 10 values):", img_np.flatten()[:10])
        img_np = (img_np * 255).astype(np.uint8)
        
        img_np = np.squeeze(img_np, axis=0) # 移除批次維度 (Batch)
        img_np = np.transpose(img_np,(0,1,2)) #轉成 (H,W,C) 
        
        if img_np.shape[2] == 3: #確認是 HWC
             img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif img_np.shape[2] == 4: #確認是 HWCA
            img_np = img_np[:,:,:3] #只取前三通道
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            print("Unsupported Image shape:", img_np.shape)
            return (0,0)


        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        print("Gray Image Shape:", gray.shape)
        print("Gray Data:",gray)
        if len(gray.shape) != 2:
             print("Error: gray image is not single channel!")
             return (0, 0)
             
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        print("Max Location:", maxLoc)
        x = int(maxLoc[0])
        y = int(maxLoc[1])
        print("x:", x, "y:", y)

        return (x, y)
NODE_CLASS_MAPPINGS = {
        "FindBrightestSpot": FindBrightestSpot
}