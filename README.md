# FindBrightestSpot

![alt text](https://i.imgur.com/7BGHoyg.jpeg)

The `FindBrightestSpot` custom node is designed to analyze an input image and determine the x and y coordinates of its brightest pixel. This node leverages the OpenCV library to efficiently perform this analysis.

It's particularly useful in ComfyUI workflows where you need to automatically identify the most luminous area within an image, enabling subsequent actions or adjustments based on that location.

---

## Functionality:

### Image Input:
The node accepts a ComfyUI image tensor as input. This should be a standard ComfyUI `IMAGE` type output from nodes like `Load Image`, or `Sampler`.

### Image Processing:
*   The image data, originally a PyTorch tensor, is converted to a NumPy array and scaled to a range of 0-255.
*   It is then transposed to have the (Height, Width, Channels) dimension order, which is suitable for OpenCV processing.
*   If the image has an alpha channel, it is removed, reducing it to 3 RGB channels.
*   Finally, the image is converted from RGB to BGR for OpenCV.
*   The image is then converted from BGR to Grayscale.

### Brightest Spot Detection:
The node utilizes the OpenCV function `cv2.minMaxLoc` to identify the minimum and maximum intensity values (min and max luminance) within the grayscale image. This function also outputs the (x, y) location of these values. The maximum value is the brightest point on the picture.

### Coordinate Output:
The node outputs two integer values:
*   `x`: The x-coordinate of the brightest pixel location.
*   `y`: The y-coordinate of the brightest pixel location.

---

## Required Libraries:

*   **OpenCV (cv2):** For image processing tasks like color conversion, finding min/max locations, and other manipulations.

    *   **Installation Command:**
        ```bash
        pip install opencv-python
        ```

*   **NumPy:** For numerical operations, particularly with arrays, which are essential for working with image data efficiently.

    *   **Installation Command:**
        ```bash
        pip install numpy
        ```

*   **Pillow (PIL):** Although it's imported, PIL isn't actually used in the final version of the code, so it's technically not required. However, it was needed for previous versions and might be useful for other image operations.

    *   **Installation Command:**
        ```bash
        pip install Pillow
        ```

*   **PyTorch (torch):** Needed for working with tensors, which is how ComfyUI represents image data, and for converting the tensors to NumPy arrays.

    *   **Installation Command:**
        You'll need to choose the appropriate PyTorch command based on your system and GPU support from [pytorch.org](https://pytorch.org/get-started/locally/).
        *   **Example (with CUDA support):**
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            ```
        *   **Example (CPU only):**
            ```bash
             pip install torch torchvision torchaudio
            ```

---

## Installation Guide:

Install this custom node in ComfyUI:

### Method : Using Git

1.  **Navigate to `custom_nodes` Directory:**
    *   Open your terminal or command prompt and navigate to the `custom_nodes` directory within your ComfyUI installation.
2.  **Clone the Repository:**
    *   Use the following command to clone the Git repository directly into your `custom_nodes` directory:
        ```bash
        git clone https://github.com/waynepimpzhang/comfyui-opencv-brightestspot.git
        ```
3.  **Restart ComfyUI:**
     *   Close and then re-open ComfyUI to refresh the list of custom nodes.
4.  **Usage:**
     *  You can find the node by searching for `FindBrightestSpot` in the ComfyUI interface. It will be in the `image/opencv` category.

---

## Node Code:

```python
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
