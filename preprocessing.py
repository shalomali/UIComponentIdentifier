import cv2
import numpy as np

def preprocess_and_extract_components(image_path, target_size=(64, 64)):
    """
    Loads an image, filters out shadows, detects individual drawn components,
    and returns them as a list of normalized tensors ready for a TensorFlow model.
    """
    # 1. Load image and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or invalid path: {image_path}")

    # 2. Adaptive Thresholding: Keeps ink lines dark, background pure white
    thresh_clean = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 7
    )
    
    # Invert to turn lines white on a black background purely for contour tracking
    thresh_inv = cv2.bitwise_not(thresh_clean)

    # 3. Find contours using RETR_EXTERNAL (grabs outer component boundaries perfectly)
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_components = []
    bounding_boxes = []

    img_h, img_w = img.shape

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Size restrictions: 
        # - Bigger than 25x25 pixels (filters out loose noise/pencil dots)
        # - Smaller than 85% of the total page dimensions (filters out full page borders)
        if (25 < w < img_w * 0.85) and (25 < h < img_h * 0.85):

            # Check for overlapping/duplicate extractions
            is_redundant = False
            for bx, by, bw, bh in bounding_boxes:
                if x >= bx and y >= by and (x+w) <= (bx+bw) and (y+h) <= (by+bh):
                    is_redundant = True
                    break

            if not is_redundant:
                # Crop from the clean, un-inverted threshold image (black drawings on white)
                cropped = thresh_clean[y:y+h, x:x+w]

                # Create a solid black background canvas (zeros)
                side = max(w, h)
                padded = np.zeros((side, side), dtype="uint8")
                
                # Invert the cropped piece right here so it becomes white drawings on a black background
                # This matches your model weights perfectly
                cropped_inverted = cv2.bitwise_not(cropped)

                # Center the inverted element onto your black canvas
                dx = (side - w) // 2
                dy = (side - h) // 2
                padded[dy:dy+h, dx:dx+w] = cropped_inverted

                # Resize and normalize cleanly
                resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
                normalized = resized.astype('float32') / 255.0
                final_tensor = np.expand_dims(normalized, axis=-1)

                extracted_components.append(final_tensor)
                bounding_boxes.append((x, y, w, h))

    return extracted_components, bounding_boxes, thresh_clean