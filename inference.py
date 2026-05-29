import numpy as np
import tensorflow as tf
# Import your preprocessing tool from the other file cleanly
from preprocessing import preprocess_and_extract_components

def run_ui_inference(full_image_path, trained_model, class_names):
    """
    Extracts components from a full page photo and prints out individual 
    classification predictions using the trained model weights.
    """
    # 1. Call the extraction pipeline from preprocessing.py
    components, bounding_boxes, _ = preprocess_and_extract_components(full_image_path)

    print(f"Detected {len(components)} potential UI elements.\n")
    print(f"{'Class Detected':<20} | {'Confidence':<10} | {'Box (x, y, w, h)'}")
    print("-" * 65)

    results = []

    # 2. Iterate and feed raw batches into the neural network
    for i, component_tensor in enumerate(components):
        # Ensure the batch dimension is explicitly formatted: (1, 64, 64, 1)
        input_batch = np.expand_dims(component_tensor, axis=0)

        # Run prediction
        predictions = trained_model.predict(input_batch, verbose=0)[0]
        
        predicted_class_idx = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_idx]
        confidence = 100 * predictions[predicted_class_idx]

        x, y, w, h = bounding_boxes[i]
        print(f"{predicted_class_name:<20} | {confidence:>8.2f}% | Box: ({x}, {y}, {w}, {h})")

        results.append({
            "class": predicted_class_name,
            "confidence": float(confidence),
            "box": (int(x), int(y), int(w), int(h))
        })

    return results