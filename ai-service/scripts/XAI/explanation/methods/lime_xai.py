import sys
sys.path.append('../../src')
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import torch
from torchvision.transforms import v2
from lime import lime_image
from model.frame import FrameModel
import torch.nn.functional as F
import numpy as np


def explain(image, transforms, label, model, custom_seg=None, visualize=False):
    """
    Generate LIME explanation for the given image.
    
    Args:
        image: Input image (numpy array, H×W×C)
        transforms: Transforms to apply to the image for model input
        label: Target class index to explain
        model: PyTorch model
        custom_seg: Custom segmentation function (optional)
        visualize: Whether to show visualization (default: False)
        
    Returns:
        A 2D numpy array representing the saliency map
    """
    # The function used by the explainer to predict the model's output
    def predict(images):
        model.eval()
        batch = torch.stack(tuple(transforms(i) for i in images), dim=0)
        logits = model(batch.to(model.device))
        if(logits.shape[1] > 1):
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits
        return probs.detach().cpu().numpy()

    # Create an explainer
    explainer = lime_image.LimeImageExplainer()

    # Set the segmentation function if it exists and compute the explanation
    if(custom_seg == None):
        explanation = explainer.explain_instance(image, predict, num_samples=2000)
    else:
        explanation = explainer.explain_instance(image, predict, segmentation_fn=custom_seg, num_samples=2000)

    # Get the image and the explanation mask
    temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=5, hide_rest=False)
    
    # Convert segments to a saliency map
    segments = explanation.segments
    
    # Create an empty saliency map
    saliency_map = np.zeros((segments.shape[0], segments.shape[1]), dtype=np.float32)
    
    # Get weights for each segment
    weights = dict(explanation.local_exp[label])
    
    # Fill in the saliency map with weights
    for segment_id, weight in weights.items():
        saliency_map[segments == segment_id] = weight
    
    # Normalize weights to range [0, 1] for visualization
    if np.max(saliency_map) != np.min(saliency_map):
        saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
    
    # If selected visualize the result
    if(visualize):
        img_boundaries = mark_boundaries(temp, mask)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_boundaries)
        plt.title("LIME Segmentation")
        plt.subplot(1, 2, 2)
        plt.imshow(saliency_map, cmap='jet')
        plt.title("LIME Saliency Map")
        plt.colorbar()
        plt.show()

    # Return the saliency map for compatibility with other XAI methods
    return saliency_map

if __name__ == "__main__":
    # Load the model
    rs_size = 224
    model = FrameModel.load_from_checkpoint("../../model/checkpoint/ff_attribution.ckpt", map_location='cuda').eval()

    # Create the transforms for inference and visualization purposes
    interpolation = 3
    inference_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(rs_size, interpolation=interpolation, antialias=False),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    visualize_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(rs_size, interpolation=interpolation, antialias=False),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # Open the image
    image = Image.open('test.jpg')
    # Apply the transformations
    visualize_image = visualize_transforms(image).permute(1, 2, 0).numpy()
    # Select the explanation label
    label = 0

    # Call the explanation method
    saliency_map = explain(visualize_image, inference_transforms, label, model, visualize=True)
    print(f"Saliency map shape: {saliency_map.shape}, dtype: {saliency_map.dtype}")