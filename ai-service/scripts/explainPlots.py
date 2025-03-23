import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def detect_tampering(gradcam_heatmap, lime_explanation, saliency_map=None,
                    gradcam_threshold=0.5, lime_threshold=0.2, saliency_threshold=0.3,
                    overlap_threshold=0.3, cluster_threshold=3,
                    plot_result=False):
    """
    Detect if an image has been tampered with using GradCAM, LIME, and Saliency explanations.
    
    Parameters:
    -----------
    gradcam_heatmap : numpy.ndarray
        GradCAM heatmap, values should be normalized between 0 and 1
    lime_explanation : numpy.ndarray
        LIME explanation mask, values should be normalized between 0 and 1
    saliency_map : numpy.ndarray, optional
        Saliency map highlighting important image regions, values should be normalized between 0 and 1
    gradcam_threshold : float, default=0.5
        Threshold above which GradCAM activation is considered significant
    lime_threshold : float, default=0.2
        Threshold above which LIME explanation is considered significant
    saliency_threshold : float, default=0.3
        Threshold above which Saliency map is considered significant
    overlap_threshold : float, default=0.3
        Required overlap ratio between explanation methods to confirm tampering
    cluster_threshold : int, default=3
        Minimum number of clusters to consider image as tampered
    plot_result : bool, default=False
        Whether to plot the results for visualization
        
    Returns:
    --------
    dict
        Dictionary containing:
        - is_tampered: boolean indicating if image is tampered
        - confidence: float between 0 and 1 indicating confidence level
        - tampered_regions: list of regions (bounding boxes) where tampering is detected
        - explanation: string explaining the decision
        - composite_mask: numpy array showing the detected tampered regions
    """
    # Ensure heatmaps are properly normalized
    if gradcam_heatmap.max() > 1.0:
        gradcam_heatmap = gradcam_heatmap / gradcam_heatmap.max()
    
    if lime_explanation.max() > 1.0:
        lime_explanation = lime_explanation / lime_explanation.max()
    
    if saliency_map is not None and saliency_map.max() > 1.0:
        saliency_map = saliency_map / saliency_map.max()
    
    # Resize explanations to match if needed
    target_shape = gradcam_heatmap.shape
    
    if lime_explanation.shape != target_shape:
        lime_explanation = cv2.resize(lime_explanation, 
                                     (target_shape[1], target_shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
    
    if saliency_map is not None and saliency_map.shape != target_shape:
        saliency_map = cv2.resize(saliency_map, 
                                 (target_shape[1], target_shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)
    
    # 1. Create binary masks from thresholds
    gradcam_mask = (gradcam_heatmap > gradcam_threshold).astype(np.uint8)
    lime_mask = (lime_explanation > lime_threshold).astype(np.uint8)
    
    # Add saliency mask if provided
    if saliency_map is not None:
        saliency_mask = (saliency_map > saliency_threshold).astype(np.uint8)
    else:
        saliency_mask = np.zeros_like(gradcam_mask)
    
    # 2. Find overlap between explanation methods
    if saliency_map is not None:
        # Three-way overlap (stronger evidence)
        three_way_overlap = gradcam_mask & lime_mask & saliency_mask
        three_way_overlap_count = np.sum(three_way_overlap)
        
        # Two-way overlaps
        gradcam_lime_overlap = gradcam_mask & lime_mask
        gradcam_saliency_overlap = gradcam_mask & saliency_mask
        lime_saliency_overlap = lime_mask & saliency_mask
        
        # Combined overlap mask (any two methods agree)
        overlap_mask = gradcam_lime_overlap | gradcam_saliency_overlap | lime_saliency_overlap
        
        # Calculate weighted overlap ratio
        total_activation = np.sum(gradcam_mask) + np.sum(lime_mask) + np.sum(saliency_mask)
        if total_activation > 0:
            # Give higher weight to three-way overlap
            overlap_ratio = (np.sum(overlap_mask) + three_way_overlap_count * 2) / total_activation
        else:
            overlap_ratio = 0
    else:
        # Original two-way overlap
        overlap_mask = gradcam_mask & lime_mask
        
        if np.sum(gradcam_mask) > 0 and np.sum(lime_mask) > 0:
            overlap_ratio = np.sum(overlap_mask) / min(np.sum(gradcam_mask), np.sum(lime_mask))
        else:
            overlap_ratio = 0
    
    # 3. Create combined mask from all available explanation methods
    if saliency_map is not None:
        combined_mask = gradcam_mask | lime_mask | saliency_mask
    else:
        combined_mask = gradcam_mask | lime_mask
    
    # 4. Analyze distribution of activations using clustering
    y_coords, x_coords = np.where(combined_mask > 0)
    
    # Default values in case no points are found
    num_clusters = 0
    cluster_sizes = []
    clustered_points = None
    
    if len(x_coords) > cluster_threshold:
        # Stack coordinates for clustering
        points = np.column_stack((x_coords, y_coords))
        
        # Apply KMeans clustering to find distinct regions
        if len(points) > cluster_threshold:
            # Determine optimal number of clusters (max 10)
            max_clusters = min(10, len(points) // 5)
            inertia = []
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(points)
                inertia.append(kmeans.inertia_)
            
            # Find elbow point
            inertia_diffs = np.diff(inertia)
            if len(inertia_diffs) > 0:
                # Use elbow method
                elbow_idx = np.argmax(np.diff(inertia_diffs)) + 1 if len(inertia_diffs) > 1 else 1
                num_clusters = elbow_idx + 1
            else:
                num_clusters = 1
            
            # Apply clustering with optimal k
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(points)
            
            # Analyze cluster sizes
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            cluster_sizes = counts
            
            # Save clustered points for visualization
            clustered_points = [(points[cluster_labels == i], i) for i in range(num_clusters)]
    
    # 5. Detect tampering regions
    tampered_regions = []
    composite_mask = np.zeros_like(gradcam_mask)
    
    if num_clusters > 0 and clustered_points:
        for points, cluster_id in clustered_points:
            if len(points) > 10:  # Minimum size to consider a valid region
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)
                
                # Calculate importance of this region
                region_mask = np.zeros_like(gradcam_mask)
                region_mask[y_min:y_max, x_min:x_max] = 1
                
                # Check how strongly each explanation method activates in this region
                gradcam_importance = np.mean(gradcam_heatmap[region_mask == 1]) if np.sum(region_mask) > 0 else 0
                lime_importance = np.mean(lime_explanation[region_mask == 1]) if np.sum(region_mask) > 0 else 0
                
                if saliency_map is not None:
                    saliency_importance = np.mean(saliency_map[region_mask == 1]) if np.sum(region_mask) > 0 else 0
                else:
                    saliency_importance = 0
                
                # Calculate average importance across methods
                if saliency_map is not None:
                    avg_importance = (gradcam_importance + lime_importance + saliency_importance) / 3
                else:
                    avg_importance = (gradcam_importance + lime_importance) / 2
                
                # Define region with importance score
                region = {
                    'x_min': int(x_min),
                    'y_min': int(y_min),
                    'x_max': int(x_max),
                    'y_max': int(y_max),
                    'size': len(points),
                    'cluster_id': int(cluster_id),
                    'importance': float(avg_importance)
                }
                
                # Add region to list
                tampered_regions.append(region)
                
                # Add to composite mask with intensity based on importance
                region_mask = np.zeros_like(gradcam_mask, dtype=float)
                region_mask[y_min:y_max, x_min:x_max] = avg_importance
                composite_mask = np.maximum(composite_mask, region_mask)
    
    # 6. Make final decision
    # Criteria for tampering
    large_cluster_count = sum(1 for size in cluster_sizes if size > 100)
    has_significant_overlap = overlap_ratio > overlap_threshold
    has_multiple_clusters = num_clusters >= cluster_threshold
    
    # Decision logic with saliency enhancement
    if saliency_map is not None:
        # When we have saliency, we have higher confidence and can be more precise
        is_tampered = (has_significant_overlap and (has_multiple_clusters or large_cluster_count >= 1))
        
        # Additional check: if three-way overlap exists, that's strong evidence
        if three_way_overlap_count > 50:  # Arbitrary threshold for significant three-way overlap
            is_tampered = True
    else:
        # Original logic
        is_tampered = (has_significant_overlap and has_multiple_clusters) or large_cluster_count >= 2
    
    # Calculate confidence score (0 to 1)
    confidence = 0.0
    if is_tampered:
        # Factors that increase confidence
        overlap_factor = min(overlap_ratio / overlap_threshold, 1.0) * 0.3
        cluster_factor = min(num_clusters / cluster_threshold, 1.0) * 0.3
        size_factor = min(sum(cluster_sizes) / (gradcam_mask.shape[0] * gradcam_mask.shape[1] * 0.1), 1.0) * 0.4
        
        # Add bonus for three-way overlap if saliency is available
        if saliency_map is not None and three_way_overlap_count > 0:
            saliency_bonus = min(three_way_overlap_count / 100, 0.2)  # Up to 0.2 bonus
            confidence = min(1.0, overlap_factor + cluster_factor + size_factor + saliency_bonus)
        else:
            confidence = overlap_factor + cluster_factor + size_factor
    else:
        # Inverse confidence for non-tampered images
        if saliency_map is not None:
            confidence = 1.0 - min(1.0, 
                                (overlap_ratio / overlap_threshold) * 0.3 + 
                                (num_clusters / max(cluster_threshold, 1)) * 0.3 +
                                (sum(cluster_sizes) / (gradcam_mask.shape[0] * gradcam_mask.shape[1] * 0.1)) * 0.4)
        else:
            confidence = 1.0 - min(1.0, 
                                (overlap_ratio / overlap_threshold) * 0.3 + 
                                (num_clusters / max(cluster_threshold, 1)) * 0.3 +
                                (sum(cluster_sizes) / (gradcam_mask.shape[0] * gradcam_mask.shape[1] * 0.1)) * 0.4)
    
    # Generate explanation
    explanation = generate_explanation(is_tampered, num_clusters, overlap_ratio, 
                                      cluster_sizes, overlap_threshold, cluster_threshold,
                                      has_saliency=saliency_map is not None)
    
    # Visualize results if requested
    if plot_result:
        visualize_results(gradcam_heatmap, lime_explanation, saliency_map, 
                         overlap_mask, composite_mask, clustered_points, is_tampered)
    
    # Return results
    results = {
        'is_tampered': is_tampered,
        'confidence': float(confidence),
        'tampered_regions': tampered_regions,
        'explanation': explanation,
        'gradcam_mask': gradcam_mask,
        'lime_mask': lime_mask,
        'overlap_mask': overlap_mask,
        'composite_mask': composite_mask,
        'num_clusters': num_clusters,
        'overlap_ratio': float(overlap_ratio)
    }
    
    # Add saliency-specific results if available
    if saliency_map is not None:
        results['saliency_mask'] = saliency_mask
        if 'three_way_overlap' in locals():
            results['three_way_overlap'] = three_way_overlap
            results['three_way_overlap_count'] = int(three_way_overlap_count)
    
    return results

def generate_explanation(is_tampered, num_clusters, overlap_ratio, 
                        cluster_sizes, overlap_threshold, cluster_threshold,
                        has_saliency=False):
    """Generate human-readable explanation for the decision."""
    if is_tampered:
        explanation = f"Image appears to be tampered. Found {num_clusters} distinct regions "
        explanation += f"with {overlap_ratio:.2f} overlap ratio (threshold: {overlap_threshold}). "
        
        if len(cluster_sizes) > 0:
            explanation += f"Largest suspicious region contains {max(cluster_sizes)} points. "
        
        if num_clusters >= cluster_threshold:
            explanation += f"Multiple suspicious regions detected ({num_clusters} >= {cluster_threshold}). "
            
        if has_saliency:
            explanation += "Detection confidence enhanced by saliency map analysis. "
    else:
        explanation = f"Image appears to be authentic. "
        if num_clusters < cluster_threshold:
            explanation += f"Found only {num_clusters} regions of interest (threshold: {cluster_threshold}). "
        if overlap_ratio < overlap_threshold:
            explanation += f"Overlap ratio ({overlap_ratio:.2f}) below threshold ({overlap_threshold}). "
        
        if len(cluster_sizes) > 0 and max(cluster_sizes) < 100:
            explanation += f"Largest region contains only {max(cluster_sizes)} points, which is not suspicious. "
            
        if has_saliency:
            explanation += "Saliency map confirmed the authenticity assessment. "
    
    return explanation

def visualize_results(gradcam_heatmap, lime_explanation, saliency_map, 
                     overlap_mask, composite_mask, clustered_points, is_tampered):
    """Visualize the tampering detection results."""
    # Determine the number of rows and columns based on available data
    rows = 2
    cols = 3
    
    if saliency_map is not None:
        rows = 3  # Add an extra row for saliency visualization
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    
    # Plot GradCAM
    axes[0, 0].imshow(gradcam_heatmap, cmap='jet')
    axes[0, 0].set_title('GradCAM Heatmap')
    axes[0, 0].axis('off')
    
    # Plot LIME
    axes[0, 1].imshow(lime_explanation, cmap='viridis')
    axes[0, 1].set_title('LIME Explanation')
    axes[0, 1].axis('off')
    
    # Plot Overlap
    axes[0, 2].imshow(overlap_mask, cmap='gray')
    axes[0, 2].set_title('Overlap Regions')
    axes[0, 2].axis('off')
    
    # Plot Composite Mask
    axes[1, 0].imshow(composite_mask, cmap='hot')
    axes[1, 0].set_title('Detected Tampered Regions')
    axes[1, 0].axis('off')
    
    # Plot Clusters
    if clustered_points:
        axes[1, 1].imshow(np.zeros_like(gradcam_heatmap), cmap='gray')
        colors = plt.cm.rainbow(np.linspace(0, 1, len(clustered_points)))
        
        for (points, cluster_id), color in zip(clustered_points, colors):
            axes[1, 1].scatter(points[:, 0], points[:, 1], c=[color], alpha=0.7, s=5)
        
        axes[1, 1].set_title('Clustered Regions')
        axes[1, 1].axis('off')
    else:
        axes[1, 1].set_title('No Clusters Found')
        axes[1, 1].axis('off')
    
    # Result Summary
    axes[1, 2].axis('off')
    result_text = "TAMPERED" if is_tampered else "AUTHENTIC"
    axes[1, 2].text(0.5, 0.5, result_text, 
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=24, 
                   color='red' if is_tampered else 'green',
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Add saliency visualization if provided
    if saliency_map is not None:
        axes[2, 0].imshow(saliency_map, cmap='magma')
        axes[2, 0].set_title('Saliency Map')
        axes[2, 0].axis('off')
        
        # Three-way overlap visualization (if exists in locals)
        if 'three_way_overlap' in locals():
            axes[2, 1].imshow(three_way_overlap, cmap='Reds')
            axes[2, 1].set_title('Three-way Overlap')
        else:
            # Just make an empty plot
            axes[2, 1].axis('off')
            
        # Add a confidence visualization
        axes[2, 2].axis('off')
        conf_text = f"Confidence: {confidence:.2%}"
        axes[2, 2].text(0.5, 0.5, conf_text,
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize=20,
                      color='black',
                      bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Example usage with sample data
if __name__ == "__main__":
    # Create synthetic data for demonstration
    # In a real-world scenario, these would come from your model's XAI outputs
    
    # Sample size
    height, width = 224, 224
    
    # Case 1: Tampered image (simulated)
    print("Case 1: Simulated tampered image with saliency")
    # Create a gradcam heatmap with a few hot spots
    gradcam_tampered = np.zeros((height, width))
    # Add a few activation regions
    gradcam_tampered[50:80, 50:80] = 0.9  # Top-left region
    gradcam_tampered[150:180, 150:180] = 0.8  # Bottom-right region
    
    # Create a matching LIME explanation
    lime_tampered = np.zeros((height, width))
    lime_tampered[45:85, 45:85] = 0.7  # Top-left region, slightly larger
    lime_tampered[145:185, 145:185] = 0.6  # Bottom-right region, slightly larger
    
    # Create a matching saliency map
    saliency_tampered = np.zeros((height, width))
    saliency_tampered[48:82, 48:82] = 0.8  # Top-left region
    saliency_tampered[148:182, 148:182] = 0.75  # Bottom-right region
    
    result_tampered = detect_tampering(
        gradcam_tampered, 
        lime_tampered, 
        saliency_tampered,
        gradcam_threshold=0.5, 
        lime_threshold=0.2,
        saliency_threshold=0.3,
        plot_result=True
    )
    
    print(f"Is image tampered: {result_tampered['is_tampered']}")
    print(f"Confidence: {result_tampered['confidence']:.2f}")
    print(f"Explanation: {result_tampered['explanation']}")
    print(f"Detected regions: {len(result_tampered['tampered_regions'])}")
    
    # Case 2: Authentic image (simulated)
    print("\nCase 2: Simulated authentic image with saliency")
    # Create a gradcam with more uniform activation
    gradcam_authentic = np.zeros((height, width))
    gradcam_authentic[100:124, 100:124] = 0.6  # Single central region
    
    # Create a matching LIME explanation
    lime_authentic = np.zeros((height, width))
    lime_authentic[95:130, 95:130] = 0.4  # Single region, slightly larger
    
    # Create a matching saliency map
    saliency_authentic = np.zeros((height, width))
    saliency_authentic[98:126, 98:126] = 0.5  # Single central region
    
    result_authentic = detect_tampering(
        gradcam_authentic, 
        lime_authentic, 
        saliency_authentic,
        gradcam_threshold=0.5, 
        lime_threshold=0.2,
        saliency_threshold=0.3,
        plot_result=True
    )
    
    print(f"Is image tampered: {result_authentic['is_tampered']}")
    print(f"Confidence: {result_authentic['confidence']:.2f}")
    print(f"Explanation: {result_authentic['explanation']}")
    print(f"Detected regions: {len(result_authentic['tampered_regions'])}")
    
    # Case 3: Using only GradCAM and LIME (without saliency)
    print("\nCase 3: Without saliency map")
    result_without_saliency = detect_tampering(
        gradcam_tampered, 
        lime_tampered,
        gradcam_threshold=0.5, 
        lime_threshold=0.2,
        plot_result=True
    )
    
    print(f"Is image tampered: {result_without_saliency['is_tampered']}")
    print(f"Confidence: {result_without_saliency['confidence']:.2f}")
    print(f"Explanation: {result_without_saliency['explanation']}")
    print(f"Detected regions: {len(result_without_saliency['tampered_regions'])}")