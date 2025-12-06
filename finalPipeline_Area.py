import numpy as np
import cv2
import torch
from scipy import ndimage
from skimage.morphology import skeletonize
from sklearn.cluster import KMeans
import torch.nn as nn
import math


def get_proposals(img, n_clusters=10):
    """Get window proposals using SIFT features and K-means clustering."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp, descriptors = sift.detectAndCompute(gray, None)
    
    if len(kp) < n_clusters:
        n_clusters = max(1, len(kp))
    
    kp_loc = np.array([k.pt for k in kp])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(kp_loc)
    centers = kmeans.cluster_centers_
    
    return centers, kp, kp_loc, cluster_labels


def get_windows(img, centers, window_size=(150, 150)):
    """Extract windows around cluster centers."""
    windows = []
    upsample = nn.Upsample(mode='bilinear', size=(224, 224))
    
    for center in centers:
        x, y = int(center[0]), int(center[1])
        
        row_s = max(0, y - window_size[0] // 2)
        row_e = min(img.shape[0], y + window_size[0] // 2)
        col_s = max(0, x - window_size[1] // 2)
        col_e = min(img.shape[1], x + window_size[1] // 2)
        
        ii = img[row_s:row_e, col_s:col_e]
        ii = cv2.medianBlur(ii, 7)
        
        window = torch.tensor(ii)
        window = upsample(window.permute(2, 0, 1).unsqueeze(0)).squeeze().permute(1, 2, 0)
        windows.append(window)
    
    return torch.stack(windows)


def process_masks(vis_img):
    """Process image using HSV color space to create crack masks."""
    hsv = cv2.cvtColor(vis_img, cv2.COLOR_RGB2HSV)
    
    # HSV range for black/dark colors (cracks)
    lower_r = np.array([0, 0, 0])
    upper_r = np.array([360, 255, 100])
    mask_r = cv2.inRange(hsv, lower_r, upper_r)
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask_r


def get_crack_edges(img_lst):
    """Get crack edges using Canny edge detection."""
    edges_lst = []
    masks_lst = []
    
    for img in img_lst:
        i = img.detach().numpy().astype(np.uint8)
        
        # Get HSV mask
        mask = process_masks(i)
        masks_lst.append(mask)
        
        # Get edges
        i_gray = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
        i_gray = cv2.medianBlur(i_gray, 1)
        edged = cv2.Canny(i_gray, 30, 200)
        edges_lst.append(edged)
    
    return edges_lst, masks_lst


def get_whole_image_crack_mask(img):
    """Get crack mask for the entire image."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crack_mask = process_masks(img_rgb)
    return crack_mask


def measure_total_crack_area(img):
    """Measure total crack area in the entire image (in pixels)."""
    crack_mask = get_whole_image_crack_mask(img)
    crack_pixels = np.sum(crack_mask > 0)
    return crack_pixels, crack_mask


def measure_crack_width_skeleton(mask):
    """Measure crack width using skeleton and distance transform."""
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(bool)
    
    skeleton = skeletonize(binary_mask)
    distance_map = ndimage.distance_transform_edt(binary_mask)
    skeleton_distances = distance_map[skeleton] * 2
    
    if len(skeleton_distances) == 0:
        return None, skeleton, distance_map
    
    measurements = {
        'min_width': np.min(skeleton_distances),
        'max_width': np.max(skeleton_distances),
        'avg_width': np.mean(skeleton_distances),
        'median_width': np.median(skeleton_distances),
        'crack_length': np.sum(skeleton),
        'crack_area': np.sum(binary_mask)
    }
    
    return measurements, skeleton, distance_map


def create_window_thumbnail(img, edge, mask, measurements, window_idx, thumb_size=200):
    """Create a thumbnail visualization for a single window."""
    # Resize all to thumbnail size
    img_thumb = cv2.resize(img, (thumb_size, thumb_size))
    edge_thumb = cv2.resize(edge, (thumb_size, thumb_size))
    mask_thumb = cv2.resize(mask, (thumb_size, thumb_size))
    
    # Convert grayscale to BGR
    edge_bgr = cv2.cvtColor(edge_thumb, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.cvtColor(mask_thumb, cv2.COLOR_GRAY2BGR)
    
    # Create heatmap
    if measurements is not None:
        distance_map = cv2.resize(measurements['distance_map'], (thumb_size, thumb_size))
        dist_normalized = (distance_map / (distance_map.max() + 1e-6) * 255).astype(np.uint8)
        dist_colored = cv2.applyColorMap(dist_normalized, cv2.COLORMAP_JET)
        
        mask_resized = cv2.resize(mask, (thumb_size, thumb_size))
        binary_mask = mask_resized.astype(bool)
        mask_3d = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
        
        heatmap = (img_thumb * 0.5 + dist_colored * 0.5 * mask_3d).astype(np.uint8)
        
        avg_width = measurements['measurements']['avg_width']
        status_text = f"W{window_idx}: {avg_width:.1f}px"
        status_color = (0, 255, 0)
    else:
        heatmap = img_thumb.copy()
        status_text = f"W{window_idx}: No crack"
        status_color = (128, 128, 128)
    
    # Create 2x2 grid for this window
    top_row = np.hstack([img_thumb, edge_bgr])
    bottom_row = np.hstack([mask_bgr, heatmap])
    grid = np.vstack([top_row, bottom_row])
    
    # Add border and label
    bordered = cv2.copyMakeBorder(grid, 25, 5, 5, 5, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(bordered, status_text, (10, 18), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    return bordered


def visualize_whole_image(img, whole_crack_mask, total_crack_area):
    """Create visualization for whole image analysis."""
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_vis = cv2.cvtColor(whole_crack_mask, cv2.COLOR_GRAY2BGR)
    
    overlay = img_rgb.copy()
    overlay[whole_crack_mask > 0] = [255, 0, 0]
    blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
    
    # Resize to reasonable size
    max_width = 1800
    if img_rgb.shape[1] > max_width // 3:
        scale = (max_width // 3) / img_rgb.shape[1]
        new_w = int(img_rgb.shape[1] * scale)
        new_h = int(img_rgb.shape[0] * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h))
        mask_vis = cv2.resize(mask_vis, (new_w, new_h))
        blended = cv2.resize(blended, (new_w, new_h))
    
    # Add text to blended
    if total_crack_area == 0:
        cv2.putText(blended, "No cracks detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(blended, f"Total Area: {total_crack_area} px", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Stack horizontally
    whole_panel = np.hstack([img_rgb, mask_vis, blended])
    
    # Add title
    whole_panel = cv2.copyMakeBorder(whole_panel, 40, 10, 10, 10, 
                                     cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(whole_panel, "Whole Image Crack Analysis", 
               (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    cv2.imshow('Whole Image Analysis', whole_panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_windows(inp_imgs, edges_lst, masks_lst, measurements_lst, n_clusters):
    """Create visualization for all window analysis."""
    
    # Create thumbnails grid for all windows
    thumb_size = 150
    thumbnails = []
    
    for i in range(n_clusters):
        thumb = create_window_thumbnail(
            inp_imgs[i], edges_lst[i], masks_lst[i], 
            measurements_lst[i], i, thumb_size
        )
        thumbnails.append(thumb)
    
    # Arrange thumbnails in a grid
    cols = 5
    rows = math.ceil(n_clusters / cols)
    
    # Pad thumbnails to fill grid
    while len(thumbnails) < rows * cols:
        blank = np.zeros_like(thumbnails[0])
        thumbnails.append(blank)
    
    # Create rows
    thumbnail_rows = []
    for r in range(rows):
        row_thumbs = thumbnails[r * cols:(r + 1) * cols]
        row = np.hstack(row_thumbs)
        thumbnail_rows.append(row)
    
    # Stack all rows
    thumbnail_grid = np.vstack(thumbnail_rows)
    
    # Add title to thumbnail grid
    thumbnail_grid = cv2.copyMakeBorder(thumbnail_grid, 40, 10, 10, 10, 
                                        cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(thumbnail_grid, f"Window Analysis - {n_clusters} Windows (Original | Edges | Mask | Heatmap)", 
               (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Resize if too large for screen
    max_height = 1000
    if thumbnail_grid.shape[0] > max_height:
        scale = max_height / thumbnail_grid.shape[0]
        new_w = int(thumbnail_grid.shape[1] * scale)
        thumbnail_grid = cv2.resize(thumbnail_grid, (new_w, max_height))
    
    cv2.imshow('Window Analysis', thumbnail_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_summary(measurements_lst, n_clusters, total_crack_area):
    """Print summary of all measurements."""
    
    # Check if total crack area is zero - exit early
    if total_crack_area == 0:
        print("\n" + "="*60)
        print("CRACK MEASUREMENT SUMMARY")
        print("="*60)
        print("\n✓ No cracks detected in the image.")
        print(f"Total windows analyzed: {n_clusters}")
        print(f"Total Crack Area: 0 pixels")
        print("\n" + "="*60)
        return
    
    valid_measurements = [m for m in measurements_lst if m is not None]

    print(f"\nTotal windows analyzed: {n_clusters}")
    print(f"Windows with cracks detected: {len(valid_measurements)}")
    
    print(f"\nTotal Crack Area: {total_crack_area} pixels")
    
    if len(valid_measurements) > 0:
        all_avg_widths = [m['measurements']['avg_width'] for m in valid_measurements]
        all_max_widths = [m['measurements']['max_width'] for m in valid_measurements]
        all_lengths = [m['measurements']['crack_length'] for m in valid_measurements]
        
        print(f"\nWIDTH STATISTICS:")
        print(f"Average Width (mean):   {np.mean(all_avg_widths):.2f} px")
        print(f"Average Width (median): {np.median(all_avg_widths):.2f} px")
        print(f"Maximum Width:          {np.max(all_max_widths):.2f} px")
        print(f"Minimum Width:          {np.min([m['measurements']['min_width'] for m in valid_measurements]):.2f} px")
        
        print(f"\nGEOMETRIC STATISTICS:")
        print(f"Total Crack Length:     {np.sum(all_lengths):.2f} px")
        
        print(f"\nWINDOW DETAILS:")
        for idx, m in enumerate(measurements_lst):
            if m is not None:
                meas = m['measurements']
                print(f"  Window {idx}: Width={meas['avg_width']:.2f} px, Length={meas['crack_length']:.2f} px")
    else:
        print("\nNo cracks detected in any window.")
    
    print("\n" + "="*60)


def main(image_path, n_clusters=20):
    """Main function to run crack detection pipeline."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    print(f"Processing: {image_path}")
    print(f"Image shape: {img.shape}")
    
    # Measure total crack area in whole image
    total_crack_area, whole_crack_mask = measure_total_crack_area(img)
    print(f"Total crack area: {total_crack_area} pixels")
    
    # If no cracks detected, show visualization and exit gracefully
    if total_crack_area == 0:
        print("\n✓ No cracks detected in the image. Showing results...")
        visualize_whole_image(img, whole_crack_mask, total_crack_area)
        print_summary([], n_clusters, total_crack_area)
        return
    
    # Get proposals using SIFT and K-means
    centers, kp, kp_loc, cluster_labels = get_proposals(img, n_clusters=n_clusters)
    print(f"Found {len(kp)} keypoints, created {len(centers)} clusters")
    
    # Extract windows
    windows = get_windows(img, centers)
    
    # Get crack edges and masks
    edges_lst, masks_lst = get_crack_edges(windows)
    
    # Measure crack dimensions
    measurements_lst = []
    for i, mask in enumerate(masks_lst):
        measurements, skeleton, distance_map = measure_crack_width_skeleton(mask)
        if measurements is not None:
            measurements_lst.append({
                'measurements': measurements,
                'skeleton': skeleton,
                'distance_map': distance_map
            })
        else:
            measurements_lst.append(None)
    
    # Two separate visualizations
    inp_imgs = [w.detach().numpy().astype(np.uint8) for w in windows]
    visualize_whole_image(img, whole_crack_mask, total_crack_area)
    visualize_windows(inp_imgs, edges_lst, masks_lst, measurements_lst, n_clusters)
    
    # Print summary
    print_summary(measurements_lst, n_clusters, total_crack_area)


if __name__ == "__main__":
    IMAGE_PATH = 'crack.jpg'  # Change to your image path
    # IMAGE_PATH = 'noCrack.png'  # Change to your image path
    N_CLUSTERS = 20
    
    main(IMAGE_PATH, n_clusters=N_CLUSTERS)