import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import cv2  # Changed from 'import cv2 as cv'
from keras.models import load_model
from website.ImageForgeryDetection.NeuralNets import initClassifier, initSegmenter
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.util import random_noise, img_as_float
import skimage.io

# Define the project root directory dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEDIA_ROOT = os.path.join(PROJECT_ROOT, 'media')
TEMP_MEDIA_PATH = os.path.join(MEDIA_ROOT, 'temp')
os.makedirs(TEMP_MEDIA_PATH, exist_ok=True)

# Define specific temporary file names using the TEMP_MEDIA_PATH
resaved_filename_base = os.path.join(TEMP_MEDIA_PATH, 'tempresaved.jpg')
luminance_gradient_filename = os.path.join(TEMP_MEDIA_PATH, 'luminance_gradient.tiff')
noise_analysis_temp_filename = os.path.join(TEMP_MEDIA_PATH, 'temp_na_resaved.jpg')
edge_detect_filename = os.path.join(TEMP_MEDIA_PATH, 'temp_edge_resaved.jpg')
ela_show_filename = os.path.join(TEMP_MEDIA_PATH, 'temp_ela_resaved.jpg')
mask_filename = os.path.join(TEMP_MEDIA_PATH, 'temp_mask.jpg')

class FID:

    def prepare_image(self, fname):
        image_size = (128, 128)
        temp_prepare_path = os.path.join(TEMP_MEDIA_PATH, 'temp_prepare.jpg')
        ela_image = self.convert_to_ela_image(fname, 90, temp_prepare_path)
        if ela_image is None:
            print("Error: Failed to prepare ELA image.")
            return None
        return np.array(ela_image.resize(image_size)).flatten() / 255.0

    def predict_result(self, fname):
        model_path = os.path.join(PROJECT_ROOT, 'ml_models', 'proposed_ela_50_casia_fidac.h5')
        model = load_model(model_path)
        class_names = ['Forged', 'Authentic'] #(index 0:Forged, index 1:Authentic)
        test_image = self.prepare_image(fname)
        if test_image is None:
            return ("Error", "Could not prepare image for prediction")
        test_image = test_image.reshape(-1, 128, 128, 3)
        y_pred = model.predict(test_image) # Probability
        y_pred_class = int(round(y_pred[0][0]))
        prediction = class_names[y_pred_class]
        confidence = f'{((1 - y_pred[0][0]) if y_pred <= 0.5 else y_pred[0][0]) * 100:0.2f}'
        return (prediction, confidence) #Return value: ('Authentic', '87.00')

    def genMask(self, file_path, save_path=None):
        # First try the standard segmenter approach
        segmenter = initSegmenter()
        weights_path = os.path.join(PROJECT_ROOT, 'ml_models', 'segmenter_weights.h5')
        segmenter.load_weights(weights_path)
        temp_mask_ela_path = os.path.join(TEMP_MEDIA_PATH, 'temp_mask_ela.jpg')
        ela_image = self.convert_to_ela_image(file_path, 90, temp_mask_ela_path)
        
        if ela_image is None:
            print("Error: Failed to generate ELA image for mask.")
            # Try advanced detection as fallback
            advanced_mask = self.generate_advanced_mask(file_path)
            if advanced_mask is not None:
                if save_path:
                    advanced_mask.save(save_path, 'JPEG')
                else:
                    advanced_mask.save(mask_filename, 'JPEG')
                return advanced_mask
            return None
        
        testimg = ela_image.resize((256, 256)).getchannel('B')
        test = np.array(testimg) / np.max(testimg)
        test = test.reshape(-1, 256, 256, 1)
        mask = segmenter.predict(test).reshape(256, 256)
        
        # If the mask is too sparse (less than 1% of pixels), try advanced detection
        if np.mean(mask) < 0.01:
            advanced_mask = self.generate_advanced_mask(file_path)
            if advanced_mask is not None:
                if save_path:
                    advanced_mask.save(save_path, 'JPEG')
                else:
                    advanced_mask.save(mask_filename, 'JPEG')
                return advanced_mask
        
        mask = (mask * 255).astype('uint8')
        mask_im = Image.fromarray(mask)
        if save_path:
            mask_im.save(save_path, 'JPEG')  # Save to the specified path
        else:
            mask_im.save(mask_filename, 'JPEG')  # Default save location
        return mask_im
    
    def generate_advanced_mask(self, file_path):
        """
        Generate a mask using advanced detection techniques when the standard
        segmenter doesn't produce good results.
        """
        try:
            # Load image
            img = cv2.imread(file_path)
            if img is None:
                return None
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. DCT Analysis
            h, w = gray.shape
            block_size = 8
            dct_scores = np.zeros((h//block_size, w//block_size), dtype=np.float32)
            
            for y in range(0, h-block_size+1, block_size):
                for x in range(0, w-block_size+1, block_size):
                    block = gray[y:y+block_size, x:x+block_size].astype(np.float32)
                    dct_block = cv2.dct(block)
                    dct_scores[y//block_size, x//block_size] = np.sum(np.abs(dct_block[4:, 4:]))
            
            # Normalize DCT scores
            dct_scores = cv2.normalize(dct_scores, None, 0, 1, cv2.NORM_MINMAX)
            
            # 2. Local noise analysis
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blur)
            
            # Calculate local noise variance in blocks
            noise_scores = np.zeros((h//block_size, w//block_size), dtype=np.float32)
            for y in range(0, h-block_size+1, block_size):
                for x in range(0, w-block_size+1, block_size):
                    block = noise[y:y+block_size, x:x+block_size]
                    noise_scores[y//block_size, x//block_size] = np.var(block)
            
            # Normalize noise scores
            noise_scores = cv2.normalize(noise_scores, None, 0, 1, cv2.NORM_MINMAX)
            
            # 3. Edge inconsistency
            edges = cv2.Canny(gray, 50, 150)
            edge_scores = np.zeros((h//block_size, w//block_size), dtype=np.float32)
            for y in range(0, h-block_size+1, block_size):
                for x in range(0, w-block_size+1, block_size):
                    block = edges[y:y+block_size, x:x+block_size]
                    edge_scores[y//block_size, x//block_size] = np.sum(block) / (block_size * block_size)
            
            # Normalize edge scores
            edge_scores = cv2.normalize(edge_scores, None, 0, 1, cv2.NORM_MINMAX)
            
            # Combine all scores - weighted sum
            combined_scores = 0.4 * dct_scores + 0.4 * noise_scores + 0.2 * edge_scores
            
            # Threshold to identify potential forgery regions
            _, binary_map = cv2.threshold(combined_scores, 0.6, 1, cv2.THRESH_BINARY)
            
            # Resize to original image size
            binary_map_resized = cv2.resize(binary_map, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Apply morphological operations to clean up the map
            kernel = np.ones((5, 5), np.uint8)
            binary_map_resized = cv2.morphologyEx(binary_map_resized, cv2.MORPH_OPEN, kernel)
            binary_map_resized = cv2.morphologyEx(binary_map_resized, cv2.MORPH_CLOSE, kernel)
            
            # Convert to PIL Image
            mask = (binary_map_resized * 255).astype(np.uint8)
            return Image.fromarray(mask)
            
        except Exception as e:
            print(f"Error in advanced mask generation: {e}")
            return None

    def convert_to_ela_image(self, path, quality, temp_save_path=resaved_filename_base):
        if not os.path.exists(path):
            print(f"Error: Input image path does not exist: {path}")
            return None
        try:
            original_image = Image.open(path).convert('RGB')
            original_image.save(temp_save_path, 'JPEG', quality=quality)
            resaved_image = Image.open(temp_save_path)
        except Exception as e:
            print(f"Error processing ELA image: {e}")
            return None
        ela_image = ImageChops.difference(original_image, resaved_image)
        extrema = ela_image.getextrema()
        max_difference = max([pix[1] for pix in extrema])
        scale = 255.0 / (max_difference if max_difference != 0 else 1)
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        return ela_image

    def show_ela(self, file_path, save_path=None):
        ela_image = self.convert_to_ela_image(file_path, 90)
        if ela_image is None:
            print("Error: Failed to generate ELA image.")
            return None
        
        # Enhance the ELA visualization
        try:
            # Convert to numpy array
            ela_array = np.array(ela_image)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better visualization
            lab = cv2.cvtColor(ela_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced_ela = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            
            # Convert back to PIL Image
            enhanced_ela_image = Image.fromarray(enhanced_ela)
            
            if save_path:
                enhanced_ela_image.save(save_path, 'JPEG')
            else:
                enhanced_ela_image.save(os.path.join(TEMP_MEDIA_PATH, 'ela_output.jpg'), 'JPEG')
            return enhanced_ela_image
        except Exception as e:
            print(f"Error enhancing ELA image: {e}, falling back to standard ELA")
            # Fall back to standard ELA if enhancement fails
            if save_path:
                ela_image.save(save_path, 'JPEG')
            else:
                ela_image.save(os.path.join(TEMP_MEDIA_PATH, 'ela_output.jpg'), 'JPEG')
            return ela_image

    def detect_edges(self, file_path, save_path=None):
        image = Image.open(file_path)
        edges = image.filter(ImageFilter.FIND_EDGES)
        if save_path:
            edges.save(save_path, 'JPEG')  # Save to the specified path
        else:
            edges.save(os.path.join(TEMP_MEDIA_PATH, 'edges_output.jpg'), 'JPEG')  # Default save location
        return edges

    # Update the copy_move_sift method to ensure it works with OpenCV 4.x
    def copy_move_sift(self, file_path, save_path=None):
        """
        Perform copy-move forgery detection using SIFT and visualize matching regions.
        """
        try:
            # Load image
            img = cv2.imread(file_path)
            if img is None:
                print("Error: Could not read image for copy-move detection.")
                return None
    
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
            # Detect SIFT keypoints and descriptors
            # For OpenCV 4.x compatibility
            try:
                sift = cv2.SIFT_create(nfeatures=5000)  # Increase number of features
            except AttributeError:
                # Fallback for older OpenCV versions
                sift = cv2.xfeatures2d.SIFT_create(nfeatures=5000)
                
            kp, des = sift.detectAndCompute(gray, None)
    
            if des is None or len(des) < 2:
                print("Not enough features found in the image.")
                return None
    
            # Match keypoints using FLANN
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)  # Increase checks for better matches
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des, des, k=2)
    
            # Filter matches and collect points
            good_matches = []
            for i, (m, n) in enumerate(matches):
                # Lower ratio for stricter matching
                if m.distance < 0.6 * n.distance and m.queryIdx != m.trainIdx:
                    # Avoid self-match (same keypoint)
                    pt1 = kp[m.queryIdx].pt
                    pt2 = kp[m.trainIdx].pt
                    
                    # Calculate distance between points
                    dist = cv2.norm(np.array(pt1) - np.array(pt2))
                    
                    # Only consider points that are far enough apart (likely copy-move)
                    # but not too far (might be false positives)
                    if 20 < dist < 300:
                        good_matches.append((pt1, pt2, m.distance/n.distance))
            
            # Sort matches by quality (lower ratio = better match)
            good_matches.sort(key=lambda x: x[2])
            
            # Take only the best matches
            good_matches = good_matches[:min(100, len(good_matches))]
            
            # Create a mask for visualization
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Draw matched keypoints on the mask
            for pt1, pt2, _ in good_matches:
                pt1 = tuple(map(int, pt1))
                pt2 = tuple(map(int, pt2))
                
                # Draw circles at matched points
                cv2.circle(mask, pt1, 10, 255, -1)
                cv2.circle(mask, pt2, 10, 255, -1)
            
            # Apply morphological operations to connect nearby points
            kernel = np.ones((15, 15), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create output visualization
            output_img = img.copy()
            
            # Draw contours on the output image
            cv2.drawContours(output_img, contours, -1, (0, 255, 0), 2)
            
            # Draw lines between matching points
            for pt1, pt2, _ in good_matches:
                pt1 = tuple(map(int, pt1))
                pt2 = tuple(map(int, pt2))
                color = (0, 0, 255)  # Red lines
                cv2.line(output_img, pt1, pt2, color, 1)
            
            # Create a heatmap overlay
            heatmap = np.zeros_like(img)
            cv2.drawContours(heatmap, contours, -1, (0, 0, 255), -1)  # Fill contours with red
            
            # Blend the heatmap with the original image
            alpha = 0.3
            output_img = cv2.addWeighted(output_img, 1 - alpha, heatmap, alpha, 0)
            
            # Add text to indicate if copy-move forgery is detected
            if len(good_matches) > 5:
                cv2.putText(output_img, "Copy-Move Detected", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Save the result
            if save_path:
                cv2.imwrite(save_path, output_img)
            else:
                cv2.imwrite(os.path.join(TEMP_MEDIA_PATH, 'sift_output.jpg'), output_img)
    
            # Convert to PIL Image and return
            output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(output_img_rgb)
    
        except Exception as e:
            print(f"Error in copy-move detection: {e}")
            return None

    def apply_na(self, file_path, save_path=None, sl=50):
        na = self.noise_analysis(file_path, 90, sl)
        if na is None:
            print("Error: Noise analysis failed.")
            return None
        if save_path:
            na.save(save_path, 'JPEG')
        else:
            na.save(os.path.join(TEMP_MEDIA_PATH, 'noise_output.jpg'), 'JPEG')
        return na

    def noise_analysis(self, file_path, quality=90, sensitivity=50):
        """
        Enhanced noise analysis using multiple techniques
        """
        try:
            # Read image using cv2
            image = cv2.imread(file_path)
            if image is None:
                print("Error: Could not read image file")
                return None
                
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # High-pass filter for noise estimation
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            high_pass = cv2.absdiff(gray, blur)
            
            # Normalize noise
            norm_noise = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)
            
            # Compute local noise in blocks
            block_size = 16
            noise_map = np.zeros_like(gray, dtype=np.float32)
            
            for y in range(0, gray.shape[0], block_size):
                for x in range(0, gray.shape[1], block_size):
                    block = norm_noise[y:y+block_size, x:x+block_size]
                    std = np.std(block)
                    noise_map[y:y+block_size, x:x+block_size] = std
            
            # Normalize and convert noise map
            noise_map = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Apply wavelet-based noise estimation for comparison
            try:
                # Convert to float for skimage functions
                img_float = img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # Estimate noise using wavelets
                sigma_est = estimate_sigma(img_float, channel_axis=-1, average_sigmas=True)
                # Scale to make it comparable to our block-based approach
                sigma_scaled = sigma_est * 100
                
                # Create a blended visualization
                heatmap1 = cv2.applyColorMap(noise_map, cv2.COLORMAP_JET)
                
                # Create a second noise map based on wavelet analysis
                wavelet_noise = np.zeros_like(gray, dtype=np.float32)
                for y in range(0, gray.shape[0], block_size):
                    for x in range(0, gray.shape[1], block_size):
                        # Add some local variation based on the global estimate
                        local_var = np.var(gray[y:y+block_size, x:x+block_size]) / 255.0
                        wavelet_noise[y:y+block_size, x:x+block_size] = sigma_scaled * (0.8 + 0.4 * local_var)
                
                # Normalize and convert wavelet noise map
                wavelet_noise = cv2.normalize(wavelet_noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                heatmap2 = cv2.applyColorMap(wavelet_noise, cv2.COLORMAP_PLASMA)
                
                # Blend the two noise visualizations
                result = cv2.addWeighted(heatmap1, 0.6, heatmap2, 0.4, 0)
            except Exception as e:
                print(f"Wavelet noise estimation failed: {e}, using only block-based approach")
                # Fall back to just the block-based approach
                result = cv2.applyColorMap(noise_map, cv2.COLORMAP_JET)
            
            # Blend with black background for better visibility
            black_bg = np.zeros_like(result)
            result = cv2.addWeighted(black_bg, 0.3, result, 0.7, 0)
            
            # Add a color bar for reference
            h, w = result.shape[:2]
            bar_width = 20
            color_bar = np.zeros((h, bar_width, 3), dtype=np.uint8)
            for i in range(h):
                value = 255 - int(255 * i / h)
                color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
                color_bar[i, :] = color
            
            # Add the color bar to the right of the image
            result = np.hstack((result, color_bar))
            
            # Convert to PIL Image
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(result_rgb)
            
            return result_image
            
        except Exception as e:
            print(f"Error in noise analysis: {e}")
            return None
    
    def ela_denoise_img(self, path, quality):
        temp_filename = resaved_filename_base
        image = Image.open(path).convert('RGB')
        image.save(temp_filename, 'JPEG', quality=quality)
        temp_image = Image.open(temp_filename)
        img = img_as_float(image)
        sigma_est = estimate_sigma(img, channel_axis=-1, average_sigmas=True)
        img_bayes = denoise_wavelet(img, method='BayesShrink', mode='soft',
                                    sigma=sigma_est, rescale_sigma=True, channel_axis=-1)
        denoised_image = Image.fromarray((img_bayes * 255).astype(np.uint8))
        return denoised_image
        
    def enhanced_predict_result(self, fname):
        """
        Enhanced prediction that combines multiple detection techniques for more accurate results.
        Uses the standard model prediction along with advanced detection methods.
        """
        # Get the basic prediction first
        basic_prediction, basic_confidence = self.predict_result(fname)
        
        # If the basic prediction is already confident (>85%), return it
        if float(basic_confidence) > 85 and basic_prediction == "Authentic":
            return (basic_prediction, basic_confidence)
            
        # Otherwise, perform advanced analysis
        try:
            # Load image
            img = cv2.imread(fname)
            if img is None:
                return (basic_prediction, basic_confidence)
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. DCT Analysis for compression artifacts
            h, w = gray.shape
            block_size = 8
            dct_scores = np.zeros((h//block_size, w//block_size), dtype=np.float32)
            
            for y in range(0, h-block_size+1, block_size):
                for x in range(0, w-block_size+1, block_size):
                    block = gray[y:y+block_size, x:x+block_size].astype(np.float32)
                    dct_block = cv2.dct(block)
                    dct_scores[y//block_size, x//block_size] = np.sum(np.abs(dct_block[4:, 4:]))
            
            # Calculate DCT inconsistency score
            dct_mean = np.mean(dct_scores)
            dct_std = np.std(dct_scores)
            dct_inconsistency = dct_std / (dct_mean + 1e-10)  # Avoid division by zero
            
            # 2. Noise analysis
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blur)
            noise_std = np.std(noise)
            noise_mean = np.mean(noise)
            noise_inconsistency = noise_std / (noise_mean + 1e-10)
            
            # 3. Edge inconsistency
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (h * w)
            
            # Combine scores into a forgery likelihood
            # These weights can be adjusted based on empirical testing
            forgery_likelihood = (0.4 * dct_inconsistency + 
                                 0.4 * noise_inconsistency + 
                                 0.2 * edge_density)
            
            # Scale to a percentage
            forgery_score = min(100, max(0, forgery_likelihood * 100))
            
            # Combine with the basic prediction
            # If basic prediction is "Forged" and advanced analysis agrees, increase confidence
            if basic_prediction == "Forged" and forgery_score > 50:
                enhanced_confidence = min(100, float(basic_confidence) + (forgery_score - 50) * 0.5)
                return ("Forged", f"{enhanced_confidence:.2f}")
            
            # If basic prediction is "Authentic" but advanced analysis disagrees strongly
            elif basic_prediction == "Authentic" and forgery_score > 70:
                # Override to "Forged" with confidence based on forgery score
                return ("Forged", f"{forgery_score:.2f}")
            
            # Otherwise, stick with the basic prediction but adjust confidence slightly
            else:
                if basic_prediction == "Forged":
                    adjusted_confidence = min(100, float(basic_confidence) + (forgery_score - 50) * 0.2)
                else:  # Authentic
                    adjusted_confidence = min(100, float(basic_confidence) + (50 - forgery_score) * 0.2)
                return (basic_prediction, f"{adjusted_confidence:.2f}")
                
        except Exception as e:
            print(f"Error in enhanced prediction: {e}")
            # Fall back to basic prediction if advanced analysis fails
            return (basic_prediction, basic_confidence)
            
    def advanced_forgery_detection(self, file_path, save_path=None):
        """
        Advanced forgery detection method that combines multiple techniques:
        1. DCT (Discrete Cosine Transform) analysis
        2. Local noise inconsistency detection
        3. Edge inconsistency analysis
        
        Returns a visualization of potential forgery regions
        """
        try:
            # Load image
            img = cv2.imread(file_path)
            if img is None:
                print("Error: Could not read image for advanced forgery detection.")
                return None
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. DCT Analysis - detect compression artifacts inconsistencies
            h, w = gray.shape
            block_size = 8  # JPEG standard block size
            dct_scores = np.zeros((h//block_size, w//block_size), dtype=np.float32)
            
            for y in range(0, h-block_size+1, block_size):
                for x in range(0, w-block_size+1, block_size):
                    block = gray[y:y+block_size, x:x+block_size].astype(np.float32)
                    dct_block = cv2.dct(block)
                    # Analyze DCT coefficients - high frequency components
                    dct_scores[y//block_size, x//block_size] = np.sum(np.abs(dct_block[4:, 4:]))
            
            # Normalize DCT scores
            dct_scores = cv2.normalize(dct_scores, None, 0, 1, cv2.NORM_MINMAX)
            
            # 2. Local noise analysis
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blur)
            
            # Calculate local noise variance in blocks
            noise_scores = np.zeros((h//block_size, w//block_size), dtype=np.float32)
            for y in range(0, h-block_size+1, block_size):
                for x in range(0, w-block_size+1, block_size):
                    block = noise[y:y+block_size, x:x+block_size]
                    noise_scores[y//block_size, x//block_size] = np.var(block)
            
            # Normalize noise scores
            noise_scores = cv2.normalize(noise_scores, None, 0, 1, cv2.NORM_MINMAX)
            
            # 3. Edge inconsistency
            edges = cv2.Canny(gray, 50, 150)
            edge_scores = np.zeros((h//block_size, w//block_size), dtype=np.float32)
            for y in range(0, h-block_size+1, block_size):
                for x in range(0, w-block_size+1, block_size):
                    block = edges[y:y+block_size, x:x+block_size]
                    edge_scores[y//block_size, x//block_size] = np.sum(block) / (block_size * block_size)
            
            # Normalize edge scores
            edge_scores = cv2.normalize(edge_scores, None, 0, 1, cv2.NORM_MINMAX)
            
            # Combine all scores - weighted sum
            combined_scores = 0.4 * dct_scores + 0.4 * noise_scores + 0.2 * edge_scores
            
            # Threshold to identify potential forgery regions
            _, binary_map = cv2.threshold(combined_scores, 0.6, 1, cv2.THRESH_BINARY)
            
            # Resize to original image size
            binary_map_resized = cv2.resize(binary_map, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Apply morphological operations to clean up the map
            kernel = np.ones((5, 5), np.uint8)
            binary_map_resized = cv2.morphologyEx(binary_map_resized, cv2.MORPH_OPEN, kernel)
            binary_map_resized = cv2.morphologyEx(binary_map_resized, cv2.MORPH_CLOSE, kernel)
            
            # Create visualization
            heatmap = np.zeros((h, w, 3), dtype=np.uint8)
            heatmap[binary_map_resized > 0] = [0, 0, 255]  # Red for potential forgery regions
            
            # Overlay on original image
            result = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
            
            # Add contours for better visibility
            binary_map_uint8 = (binary_map_resized * 255).astype(np.uint8)
            contours, _ = cv2.findContours(binary_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            
            # Save the result
            if save_path:
                cv2.imwrite(save_path, result)
            else:
                output_path = os.path.join(TEMP_MEDIA_PATH, 'advanced_detection.jpg')
                cv2.imwrite(output_path, result)
            
            # Convert to PIL Image and return
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result_rgb)
            
        except Exception as e:
            print(f"Error in advanced forgery detection: {e}")
            return None
            
    def frequency_analysis(self, file_path, save_path=None):
        """
        Performs frequency domain analysis to detect inconsistencies in the image
        that might indicate manipulation.
        """
        try:
            # Load image
            img = cv2.imread(file_path)
            if img is None:
                print("Error: Could not read image for frequency analysis.")
                return None
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply DFT (Discrete Fourier Transform)
            dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            # Calculate magnitude spectrum
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
            
            # Normalize for visualization
            magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Apply colormap for better visualization
            magnitude_colored = cv2.applyColorMap(magnitude_spectrum, cv2.COLORMAP_JET)
            
            # Create a side-by-side comparison
            h, w = gray.shape
            result = np.zeros((h, w*2, 3), dtype=np.uint8)
            result[:, :w] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            result[:, w:] = magnitude_colored
            
            # Add dividing line
            cv2.line(result, (w, 0), (w, h), (255, 255, 255), 2)
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, "Original", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(result, "Frequency Domain", (w+10, 30), font, 1, (255, 255, 255), 2)
            
            # Save the result
            if save_path:
                cv2.imwrite(save_path, result)
            else:
                output_path = os.path.join(TEMP_MEDIA_PATH, 'frequency_analysis.jpg')
                cv2.imwrite(output_path, result)
            
            # Convert to PIL Image and return
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result_rgb)
            
        except Exception as e:
            print(f"Error in frequency analysis: {e}")
            return None