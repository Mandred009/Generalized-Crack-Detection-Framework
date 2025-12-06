import os
import numpy as np
import pandas as pd
import cv2
import torch
import random

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend for interactive display
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms, utils, models
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans


def get_proposals(img, n_clusters=10):
    """
    Get window proposals using SIFT features and K-means clustering.
    
    Args:
        img: Input image (BGR format)
        n_clusters: Number of clusters for K-means
        
    Returns:
        centers: Cluster centers
        kp: SIFT keypoints
        kp_loc: Keypoint locations
        cluster_labels: Cluster labels for each keypoint
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp, descriptors = sift.detectAndCompute(gray, None)
    
    # Extract keypoint locations
    kp_loc = np.array([k.pt for k in kp])
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(kp_loc)
    centers = kmeans.cluster_centers_
    
    return centers, kp, kp_loc, cluster_labels


def visual_kmeans_sift(centers, kp, kp_loc, cluster_labels, img, n_clusters=10):
    """
    Visualize SIFT features and K-means clustering results.
    
    Args:
        centers: Cluster centers
        kp: SIFT keypoints
        kp_loc: Keypoint locations
        cluster_labels: Cluster labels
        img: Input image
        n_clusters: Number of clusters
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    # Display original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Display SIFT keypoints
    img_with_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    axes[0, 1].imshow(img_with_kp)
    axes[0, 1].set_title(f'SIFT Keypoints ({len(kp)} points)')
    axes[0, 1].axis('off')
    
    # Plot clustered keypoints
    axes[1, 0].scatter(kp_loc[:, 0], kp_loc[:, 1], c=cluster_labels, cmap='viridis', s=20)
    axes[1, 0].scatter(centers[:, 0], centers[:, 1],
                      c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[1, 0].set_title('Clustered Keypoints')
    axes[1, 0].set_xlabel('X coordinate')
    axes[1, 0].set_ylabel('Y coordinate')
    axes[1, 0].invert_yaxis()
    axes[1, 0].legend()
    
    # Create overlay image
    overlay_img = img.copy()
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        cluster_points = kp_loc[cluster_labels == i]
        color = (colors[i][:3] * 255).astype(int).tolist()
        for point in cluster_points:
            cv2.circle(overlay_img, tuple(point.astype(int)), 3, color[::-1], -1)
        
        # Draw cluster centers
        center = centers[i].astype(int)
        cv2.drawMarker(overlay_img, tuple(center), (255, 0, 0),
                      markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(overlay_img, str(i), tuple(center), 
                   fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                   color=(255, 0, 0), thickness=2)
    
    axes[1, 1].imshow(overlay_img)
    axes[1, 1].set_title('Clusters Overlaid on Image')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Displayed clustering visualization")


def get_windows(img, centers, window_size=(150, 150)):
    """
    Extract windows around cluster centers.
    
    Args:
        img: Input image
        centers: Cluster centers
        window_size: Size of windows to extract (height, width)
        
    Returns:
        Tensor of extracted and resized windows
    """
    windows = []
    upsample = nn.Upsample(mode='bilinear', size=(224, 224))
    
    for center in list(centers):
        x = int(center[0])
        y = int(center[1])
        
        row_s = max(0, y - window_size[0] // 2)
        row_e = min(img.shape[0], y + window_size[0] // 2)
        
        col_s = max(0, x - window_size[1] // 2)
        col_e = min(img.shape[1], x + window_size[1] // 2)
        
        ii = img[row_s:row_e, col_s:col_e]
        ii = cv2.medianBlur(ii, 7)
        
        window = torch.tensor(ii)
        window = upsample(window.permute(2, 0, 1).unsqueeze(0)).squeeze().permute(1, 2, 0)
        windows.append(window)
    
    out = torch.stack(windows)
    return out


def process_masks(vis_img):
    """
    Process image using HSV color space to create masks.
    
    Args:
        vis_img: Input image (RGB format)
        
    Returns:
        Masked RGB image and grayscale image
    """
    hsv = cv2.cvtColor(vis_img, cv2.COLOR_RGB2HSV)
    
    # HSV range for black
    lower_r = np.array([0, 0, 0])
    upper_r = np.array([360, 255, 100])
    mask_r = cv2.inRange(hsv, lower_r, upper_r)
    
    vis_img_masked_r = cv2.bitwise_and(vis_img, vis_img, mask=mask_r)
    vis_img_gray_r = cv2.cvtColor(vis_img_masked_r, cv2.COLOR_RGB2GRAY)
    
    return vis_img_masked_r, vis_img_gray_r


def get_crack_intensity(contours):
    """
    Calculate total crack intensity from contours.
    
    Args:
        contours: List of contours
        
    Returns:
        Total area of all contours
    """
    area_sum = 0
    for i in contours:
        area_sum += cv2.contourArea(i)
    return area_sum


def get_contours(img_lst):
    """
    Detect contours in a list of images.
    
    Args:
        img_lst: List of images
        
    Returns:
        Input images, output images with contours, and crack intensities
    """
    contours_lst = []
    out_imgs = []
    inp_imgs = []
    areas = []
    
    for img in img_lst:
        i = img.detach().numpy()
        i = cv2.medianBlur(i, 1)
        edged = cv2.Canny(i, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        aa = i.copy()
        inp_imgs.append(i)
        cv2.drawContours(aa, contours, -1, (0, 255, 255), 3)
        
        contours_lst.append(contours)
        out_imgs.append(aa)
        areas.append(get_crack_intensity(contours))
    
    return inp_imgs, out_imgs, areas


def visualize_detections(inp, out, inten, NO=10):
    """
    Visualize crack detection results.
    
    Args:
        inp: Input images
        out: Output images with contours
        inten: Crack intensities
        NO: Number of images to visualize
    """
    fig, axes = plt.subplots(NO, 3, figsize=(18, 50))
    
    for i in range(NO):
        axes[i, 0].imshow(inp[i])
        axes[i, 0].set_title(f'Input Image: Center No {i}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(out[i])
        axes[i, 1].set_title(f'Contour: Intensity {inten[i]:.2f} pixel^2')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(process_masks(inp[i])[1], cmap='gray')
        axes[i, 2].set_title('HSV Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Displayed detection results")


def main(image_path, n_clusters=20):
    """
    Main function to run crack detection pipeline.
    
    Args:
        image_path: Path to input image
        n_clusters: Number of clusters for detection
    """
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Image shape: {img.shape}")
    print(f"Number of clusters: {n_clusters}")
    
    # Get proposals
    centers, kp, kp_loc, cluster_labels = get_proposals(img, n_clusters=n_clusters)
    
    # Visualize clustering
    visual_kmeans_sift(centers, kp, kp_loc, cluster_labels, img, n_clusters=n_clusters)
    
    # Extract windows
    windows = get_windows(img, centers)
    
    # Get contours and intensities
    inp_imgs, out_imgs, intensities = get_contours(windows)
    
    # Visualize detections
    visualize_detections(inp_imgs, out_imgs, intensities, NO=n_clusters)
    
    # Print summary
    print("\nCrack Detection Summary:")
    print(f"Total regions analyzed: {len(intensities)}")
    print(f"Average crack intensity: {np.mean(intensities):.2f} pixel^2")
    print(f"Maximum crack intensity: {np.max(intensities):.2f} pixel^2")
    print(f"Minimum crack intensity: {np.min(intensities):.2f} pixel^2")


if __name__ == "__main__":
    # Configuration
    IMAGE_PATH = "crack.jpg"  # Change this to your image path
    N_CLUSTERS = 20
    
    # Run crack detection
    main(IMAGE_PATH, n_clusters=N_CLUSTERS)