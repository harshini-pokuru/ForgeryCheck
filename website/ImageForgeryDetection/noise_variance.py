import sys
import math
import numpy as np
import cv2

from PIL import Image
from scipy import signal
from sklearn.cluster import KMeans
from skimage.restoration import estimate_sigma

def estimate_noise(I):
    H, W = I.shape

    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(signal.convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

    return sigma

def estimate_noise_wavelet(I):
    """
    Estimate noise using wavelet-based method from skimage
    More accurate for natural images
    """
    return estimate_sigma(I, multichannel=False, average_sigmas=True)

def estimate_noise_pca(I):
    """
    PCA-based noise estimation
    Better for textured regions
    """
    # Convert to float
    I_float = I.astype(np.float32) / 255.0
    
    # Create patches
    patches = []
    step = 4
    for i in range(0, I.shape[0] - 7, step):
        for j in range(0, I.shape[1] - 7, step):
            patch = I_float[i:i+8, j:j+8]
            patches.append(patch.flatten())
    
    if len(patches) < 10:  # Not enough patches
        return 0
    
    patches = np.array(patches)
    
    # Compute covariance matrix
    cov = np.cov(patches, rowvar=False)
    
    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    
    # Noise variance is estimated as the smallest eigenvalue
    noise_var = max(0, np.min(eigenvalues))
    
    return np.sqrt(noise_var)

def detect(input, blockSize=32):
    # Load image and convert to grayscale
    try:
        im = Image.open(input)
        # Convert to binary (1-bit) mode
        im_binary = im.convert('1')
        # Also create a grayscale version for additional analysis
        im_gray = im.convert('L')
        
        # Convert to numpy arrays
        img_binary = np.array(im_binary)
        img_gray = np.array(im_gray)
    except Exception as e:
        print(f"Error loading image: {e}")
        return False

    blocks_binary = []
    blocks_gray = []

    imgwidth, imgheight = im.size

    # break up image into NxN blocks, N = blockSize
    for i in range(0, imgheight, blockSize):
        for j in range(0, imgwidth, blockSize):
            # Ensure we don't go out of bounds
            if i + blockSize <= imgheight and j + blockSize <= imgwidth:
                box = (j, i, j+blockSize, i+blockSize)
                # Binary blocks
                b_binary = im_binary.crop(box)
                a_binary = np.asarray(b_binary).astype(int)
                blocks_binary.append(a_binary)
                
                # Grayscale blocks
                b_gray = im_gray.crop(box)
                a_gray = np.asarray(b_gray).astype(int)
                blocks_gray.append(a_gray)

    # Original method: Laplacian noise estimation on binary image
    variances_laplacian = []
    for block in blocks_binary:
        variances_laplacian.append([estimate_noise(block)])

    # Additional method 1: Wavelet-based noise estimation on grayscale
    variances_wavelet = []
    for block in blocks_gray:
        variances_wavelet.append([estimate_noise_wavelet(block)])
    
    # Additional method 2: PCA-based noise estimation on grayscale
    variances_pca = []
    for block in blocks_gray:
        variances_pca.append([estimate_noise_pca(block)])
    
    # Combine all features for better clustering
    combined_features = np.hstack((
        np.array(variances_laplacian),
        np.array(variances_wavelet),
        np.array(variances_pca)
    ))
    
    # Apply K-means clustering with combined features
    kmeans = KMeans(n_clusters=2, random_state=0).fit(combined_features)
    centers = kmeans.cluster_centers_
    
    # Calculate distances between cluster centers for each method
    dist_laplacian = abs(centers[0][0] - centers[1][0])
    dist_wavelet = abs(centers[0][1] - centers[1][1])
    dist_pca = abs(centers[0][2] - centers[1][2])
    
    # Weighted decision based on all methods
    threshold_laplacian = 0.4
    threshold_wavelet = 0.1
    threshold_pca = 0.05
    
    # Count how many methods indicate forgery
    forgery_count = 0
    if dist_laplacian > threshold_laplacian:
        forgery_count += 1
    if dist_wavelet > threshold_wavelet:
        forgery_count += 1
    if dist_pca > threshold_pca:
        forgery_count += 1
    
    # If at least 2 methods indicate forgery, consider it forged
    return forgery_count >= 2
