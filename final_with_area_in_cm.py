import numpy as np
import cv2
import torch
from scipy import ndimage
from skimage.morphology import skeletonize
from sklearn.cluster import KMeans
import torch.nn as nn
import math


def calibrate_with_checkerboard(gray, checker_rows=4, checker_cols=4, square_size=3.0):
    """
    Detect checkerboard and compute calibration parameters.
    
    Args:
        gray: Grayscale image
        checker_rows: Number of INNER corners rows (for NxN grid, use N-1)
        checker_cols: Number of INNER corners columns (for NxN grid, use N-1)
        square_size: Size of each square in cm
        
    Returns:
        checker_contour: Convex hull of checkerboard corners
        pixels_per_cm: Calibration factor
        calibrated: Success flag
        exclusion_mask: Mask to exclude checkerboard region from crack detection
    """
    pattern_size = (checker_cols, checker_rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    
    if not ret:
        print("Warning: Checkerboard not detected. Measurements will be in pixels.")
        return None, None, False, None
    
    # Refine corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    image_points = corners_refined.reshape(-1, 2)
    
    # Calculate pixels per cm from adjacent corners
    pixel_dist = np.linalg.norm(image_points[0] - image_points[1])
    pixels_per_cm = pixel_dist / square_size
    
    # Create convex hull around checkerboard
    checker_hull = cv2.convexHull(image_points.astype(np.float32))
    checker_contour = checker_hull.reshape(-1, 1, 2).astype(np.float32)
    
    # Create exclusion mask for checkerboard region
    exclusion_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(exclusion_mask, checker_contour.astype(np.int32), 255)
    
    # Expand the mask to cover the outer squares
    expansion_pixels = int(pixel_dist * 2.2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                       (expansion_pixels, expansion_pixels))
    exclusion_mask = cv2.dilate(exclusion_mask, kernel, iterations=1)
    
    print(f"Calibration successful: {pixels_per_cm:.2f} pixels/cm")
    print(f"Checkerboard region expanded by {expansion_pixels} pixels")
    print("Checkerboard region will be excluded from crack detection")
    
    return checker_contour, pixels_per_cm, True, exclusion_mask


def get_proposals(img, n_clusters=10, exclusion_mask=None):
    """
    Get window proposals using SIFT features and K-means clustering.
    
    Args:
        img: Input image (BGR format)
        n_clusters: Number of clusters for K-means
        exclusion_mask: Optional mask to exclude regions from SIFT detection
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # Create mask for SIFT detection (inverse of exclusion mask)
    sift_mask = None
    if exclusion_mask is not None:
        sift_mask = cv2.bitwise_not(exclusion_mask)
    
    sift = cv2.SIFT_create()
    kp, descriptors = sift.detectAndCompute(gray, mask=sift_mask)
    
    if len(kp) < n_clusters:
        n_clusters = max(1, len(kp))
    
    kp_loc = np.array([k.pt for k in kp])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(kp_loc)
    centers = kmeans.cluster_centers_
    
    return centers, kp, kp_loc, cluster_labels


def get_windows(img, centers, window_size=(150, 150), exclusion_mask=None):
    """
    Extract windows around cluster centers.
    
    Args:
        img: Input image
        centers: Cluster centers
        window_size: Size of windows to extract (height, width)
        exclusion_mask: Optional mask to extract corresponding window regions
    """
    windows = []
    exclusion_windows = []
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
        
        # Extract corresponding exclusion mask window if provided
        if exclusion_mask is not None:
            mask_window = exclusion_mask[row_s:row_e, col_s:col_e]
            mask_window_tensor = torch.tensor(mask_window).unsqueeze(0).unsqueeze(0).float()
            mask_window_resized = torch.nn.functional.interpolate(
                mask_window_tensor, size=(224, 224), mode='nearest'
            ).squeeze().numpy().astype(np.uint8)
            exclusion_windows.append(mask_window_resized)
        else:
            exclusion_windows.append(None)
    
    return torch.stack(windows), exclusion_windows


def process_masks(vis_img, exclusion_mask=None):
    """Process image using HSV color space to create crack masks."""
    hsv = cv2.cvtColor(vis_img, cv2.COLOR_RGB2HSV)
    
    # HSV range for black/dark colors (cracks)
    lower_r = np.array([0, 0, 0])
    upper_r = np.array([360, 255, 100])
    mask_r = cv2.inRange(hsv, lower_r, upper_r)
    
    # Exclude checkerboard region if provided
    if exclusion_mask is not None:
        mask_r = cv2.bitwise_and(mask_r, cv2.bitwise_not(exclusion_mask))
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask_r


def get_crack_edges(img_lst, exclusion_windows):
    """Get crack edges using Canny edge detection."""
    edges_lst = []
    masks_lst = []
    
    for idx, img in enumerate(img_lst):
        i = img.detach().numpy().astype(np.uint8)
        exclusion_mask = exclusion_windows[idx]
        
        # Get HSV mask
        mask = process_masks(i, exclusion_mask)
        masks_lst.append(mask)
        
        # Get edges
        i_gray = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
        i_gray = cv2.medianBlur(i_gray, 1)
        edged = cv2.Canny(i_gray, 30, 200)
        
        # Exclude checkerboard region from edges if provided
        if exclusion_mask is not None:
            edged = cv2.bitwise_and(edged, cv2.bitwise_not(exclusion_mask))
        
        edges_lst.append(edged)
    
    return edges_lst, masks_lst


def get_whole_image_crack_mask(img, exclusion_mask=None):
    """Get crack mask for the entire image."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crack_mask = process_masks(img_rgb, exclusion_mask)
    return crack_mask


def measure_total_crack_area(img, exclusion_mask=None, pixels_per_cm=None):
    """Measure total crack area in the entire image."""
    crack_mask = get_whole_image_crack_mask(img, exclusion_mask)
    crack_pixels = np.sum(crack_mask > 0)
    
    # Determine units and scale
    unit = "px"
    scale = 1.0
    if pixels_per_cm is not None:
        scale = 1.0 / (pixels_per_cm ** 2)
        unit = "cm²"
    
    total_area = crack_pixels * scale
    
    return total_area, crack_mask, unit


def measure_crack_width_skeleton(mask, pixels_per_cm=None):
    """Measure crack width using skeleton and distance transform."""
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(bool)
    
    skeleton = skeletonize(binary_mask)
    distance_map = ndimage.distance_transform_edt(binary_mask)
    skeleton_distances = distance_map[skeleton] * 2
    
    if len(skeleton_distances) == 0:
        return None, skeleton, distance_map
    
    # Determine units and scale
    unit = "px"
    scale = 1.0
    if pixels_per_cm is not None:
        scale = 1.0 / pixels_per_cm
        unit = "cm"
    
    measurements = {
        'min_width': np.min(skeleton_distances) * scale,
        'max_width': np.max(skeleton_distances) * scale,
        'avg_width': np.mean(skeleton_distances) * scale,
        'median_width': np.median(skeleton_distances) * scale,
        'crack_length': np.sum(skeleton) * scale,
        'crack_area': np.sum(binary_mask) * (scale ** 2),
        'unit': unit
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
        unit = measurements['measurements']['unit']
        status_text = f"W{window_idx}: {avg_width:.2f}{unit}"
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


def visualize_whole_image(img, whole_crack_mask, total_crack_area, unit, exclusion_mask=None):
    """Create visualization for whole image analysis."""
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_vis = cv2.cvtColor(whole_crack_mask, cv2.COLOR_GRAY2BGR)
    
    overlay = img_rgb.copy()
    overlay[whole_crack_mask > 0] = [255, 0, 0]
    
    # Show exclusion mask if provided
    if exclusion_mask is not None:
        overlay[exclusion_mask > 0] = [0, 255, 255]  # Cyan for excluded region
    
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
        cv2.putText(blended, f"Total Area: {total_crack_area:.2f} {unit}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if exclusion_mask is not None:
        cv2.putText(blended, "Cyan = Excluded Region", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
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


def print_summary(measurements_lst, n_clusters, total_crack_area, area_unit):
    """Print summary of all measurements."""
    
    # Check if total crack area is zero - exit early
    if total_crack_area == 0:
        print("\nNo cracks detected in the image.")
        print(f"Total windows analyzed: {n_clusters}")
        print(f"Total Crack Area: 0 {area_unit}")
        return
    
    valid_measurements = [m for m in measurements_lst if m is not None]

    print(f"\nTotal windows analyzed: {n_clusters}")
    print(f"Windows with cracks detected: {len(valid_measurements)}")
    
    print(f"\nTotal Crack Area: {total_crack_area:.2f} {area_unit}")
    
    if len(valid_measurements) > 0:
        unit = valid_measurements[0]['measurements']['unit']
        
        all_avg_widths = [m['measurements']['avg_width'] for m in valid_measurements]
        all_max_widths = [m['measurements']['max_width'] for m in valid_measurements]
        all_lengths = [m['measurements']['crack_length'] for m in valid_measurements]
        
        print(f"\nWIDTH STATISTICS:")
        print(f"Average Width (mean):   {np.mean(all_avg_widths):.3f} {unit}")
        print(f"Average Width (median): {np.median(all_avg_widths):.3f} {unit}")
        print(f"Maximum Width:          {np.max(all_max_widths):.3f} {unit}")
        print(f"Minimum Width:          {np.min([m['measurements']['min_width'] for m in valid_measurements]):.3f} {unit}")
        
        print(f"\nGEOMETRIC STATISTICS:")
        print(f"Total Crack Length:     {np.sum(all_lengths):.2f} {unit}")
        
        print(f"\nWINDOW DETAILS:")
        for idx, m in enumerate(measurements_lst):
            if m is not None:
                meas = m['measurements']
                print(f"  Window {idx}: Width={meas['avg_width']:.3f} {unit}, Length={meas['crack_length']:.2f} {unit}")
    else:
        print("\nNo cracks detected in any window.")
    


def main(image_path, n_clusters=20, checker_rows=4, checker_cols=4, square_size=3.0):
    """Main function to run crack detection pipeline with calibration."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert to grayscale for checkerboard detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calibrate with checkerboard and get exclusion mask
    checker_contour, pixels_per_cm, calibrated, exclusion_mask = calibrate_with_checkerboard(
        gray, checker_rows, checker_cols, square_size
    )
    
    # Measure total crack area in whole image
    total_crack_area, whole_crack_mask, area_unit = measure_total_crack_area(
        img, exclusion_mask, pixels_per_cm
    )
    print(f"Total crack area: {total_crack_area:.2f} {area_unit}")
    
    # If no cracks detected, show visualization and exit gracefully
    if total_crack_area == 0:
        print("\n✓ No cracks detected in the image. Showing results...")
        visualize_whole_image(img, whole_crack_mask, total_crack_area, area_unit, exclusion_mask)
        print_summary([], n_clusters, total_crack_area, area_unit)
        return
    
    # Get proposals using SIFT and K-means
    centers, kp, kp_loc, cluster_labels = get_proposals(img, n_clusters=n_clusters, 
                                                         exclusion_mask=exclusion_mask)
    print(f"Found {len(kp)} keypoints, created {len(centers)} clusters")
    
    # Extract windows
    windows, exclusion_windows = get_windows(img, centers, exclusion_mask=exclusion_mask)
    
    # Get crack edges and masks
    edges_lst, masks_lst = get_crack_edges(windows, exclusion_windows)
    edges_lst, masks_lst = get_crack_edges(windows, exclusion_windows)
    
    # Measure crack dimensions
    measurements_lst = []
    for i, mask in enumerate(masks_lst):
        measurements, skeleton, distance_map = measure_crack_width_skeleton(mask, pixels_per_cm)
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
    visualize_whole_image(img, whole_crack_mask, total_crack_area, area_unit, exclusion_mask)
    visualize_windows(inp_imgs, edges_lst, masks_lst, measurements_lst, n_clusters)
    
    # Print summary
    print_summary(measurements_lst, n_clusters, total_crack_area, area_unit)


if __name__ == "__main__":
    # Configuration
    IMAGE_PATH = 'crackC.png'  # Change to your image path
    N_CLUSTERS = 20
    CHECKER_ROWS = 4  # Number of INNER corners (for 5x5 grid, use 4)
    CHECKER_COLS = 4
    SQUARE_SIZE = 3.0  # cm
    
    main(IMAGE_PATH, n_clusters=N_CLUSTERS, checker_rows=CHECKER_ROWS, 
         checker_cols=CHECKER_COLS, square_size=SQUARE_SIZE)