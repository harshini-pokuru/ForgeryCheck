import os
import copyreg
import cv2 as cv
import numpy as np
from multiprocessing import Pool
import time

def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)
copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


class CopyMoveSIFT: 
    resize_percentage = 100
    # Further increase the distance threshold to allow even more matches
    max_dist = 250  # Increased from 200
    
    def __init__(self, path, output_path=None):
        # Validate input path
        if not path or not os.path.exists(path):
            print(f"Error: Could not read image for copy-move detection. Path: {path}")
            # Create a blank image with error message
            blank_img = np.zeros((500, 500, 3), dtype=np.uint8)
            cv.putText(blank_img, "Error: Image not found", (50, 250), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Save the error image
            if output_path:
                cv.imwrite(output_path, blank_img)
            return

        # Default output path if none provided
        self.output_path = output_path or os.path.join(os.getcwd(), 'media', 'tempresaved.jpg')

        try:
            img_gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
            img_rgb = cv.imread(path, cv.IMREAD_COLOR)

            if img_gray is None or img_rgb is None:
                print(f"Error: Could not read image at {path}")
                # Create a blank image with error message
                blank_img = np.zeros((500, 500, 3), dtype=np.uint8)
                cv.putText(blank_img, "Error: Could not read image", (50, 250), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Save the error image
                cv.imwrite(self.output_path, blank_img)
                return

            # Add a size limit to prevent processing very large images
            max_dimension = 1000  # Limit the maximum dimension to 1000 pixels
            h, w = img_gray.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                self.resize_percentage = scale * 100
            
            self.img_gray = cv.resize(img_gray, (
                int(img_gray.shape[1] * self.resize_percentage / 100),
                int(img_gray.shape[0] * self.resize_percentage / 100)))
            self.img_rgb = cv.resize(img_rgb, (
                int(img_rgb.shape[1] * self.resize_percentage / 100), 
                int(img_rgb.shape[0] * self.resize_percentage / 100)))

            # Try to use SIFT
            try:
                # Further increase the number of features detected by SIFT
                sift = cv.SIFT_create(nfeatures=10000)  # Increased from 5000
            except AttributeError:
                try:
                    sift = cv.xfeatures2d.SIFT_create(nfeatures=10000)  # Increased from 5000
                except:
                    # If SIFT is not available, create an error image
                    print("SIFT algorithm not available in this OpenCV build")
                    cv.putText(self.img_rgb, "SIFT not available in OpenCV", (50, 50), 
                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv.imwrite(self.output_path, self.img_rgb)
                    return

            self.keypoints_sift, self.descriptors = sift.detectAndCompute(self.img_gray, None)
            
            if len(self.keypoints_sift) > 0:
                # Use a smaller number of processes to avoid potential issues
                num_processes = min(4, os.cpu_count() or 1)
                pool = Pool(processes=num_processes)
                matched_pts = pool.map(self.apply_sift, np.array_split(range(len(self.descriptors)), min(num_processes, len(self.descriptors))))
                pool.close()
                self.draw(matched_pts)
            else:
                print("No keypoints detected in the image")
                # Create a copy of the original image with a message
                cv.putText(self.img_rgb, "No matching regions found", (50, 50), 
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.imwrite(self.output_path, self.img_rgb)
        except Exception as e:
            print(f"Error in SIFT processing: {e}")
            # Create a copy with error message
            if 'img_rgb' in locals() and img_rgb is not None:
                cv.putText(img_rgb, f"Error: {str(e)}", (50, 50), 
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.imwrite(self.output_path, img_rgb)
            else:
                # Create a blank image with error message
                blank_img = np.zeros((500, 500, 3), dtype=np.uint8)
                cv.putText(blank_img, f"Error: {str(e)}", (50, 250), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv.imwrite(self.output_path, blank_img)

    def compare_keypoint(self, descriptor1, descriptor2):
        return np.linalg.norm(descriptor1 - descriptor2)

    def apply_sift(self, in_vector):
        out_point_list = []
        # Further increase the number of keypoints to compare
        max_keypoints = min(3000, len(self.keypoints_sift))
        
        for index_dis in in_vector:
            if index_dis >= max_keypoints:
                continue
                
            # Further increase the number of comparisons per keypoint
            max_comparisons = min(300, len(self.keypoints_sift) - index_dis - 1)
            for i in range(min(max_comparisons, len(self.keypoints_sift) - index_dis - 1)):
                index_ic = index_dis + 1 + i
                
                point1_x = int(round(self.keypoints_sift[index_dis].pt[0]))
                point1_y = int(round(self.keypoints_sift[index_dis].pt[1]))
                point2_x = int(round(self.keypoints_sift[index_ic].pt[0]))
                point2_y = int(round(self.keypoints_sift[index_ic].pt[1]))
                
                # Skip if points are the same
                if point1_x == point2_x and point1_y == point2_y:
                    continue

                dist = self.compare_keypoint(self.descriptors[index_dis], self.descriptors[index_ic])

                if dist < self.max_dist:
                    out_point_list.append([
                        round(self.keypoints_sift[index_dis].pt[0]), 
                        round(self.keypoints_sift[index_dis].pt[1]),
                        round(self.keypoints_sift[index_ic].pt[0]), 
                        round(self.keypoints_sift[index_ic].pt[1])
                    ])

                # Increase the limit for matches to show more lines
                if len(out_point_list) > 1000:  # Increased from 500
                    break
            
            # Increase the early exit threshold
            if len(out_point_list) > 1000:  # Increased from 500
                break

        if out_point_list:
            return out_point_list
        return None

    def draw(self, matched_pts):
        has_matches = False
        for points in matched_pts:
            if points is None:
                continue
                
            has_matches = True
            for in_points in points:
                # Draw red circle for first point - increased radius from 4 to 8
                cv.circle(self.img_rgb, (in_points[0], in_points[1]),
                        6, (0, 0, 255), -1)

                # Draw blue circle for second point - increased radius from 4 to 8
                cv.circle(self.img_rgb, (in_points[2], in_points[3]),
                        6, (255, 0, 0), -1)

                # Draw green line connecting the points - increased thickness from 1 to 2
                cv.line(self.img_rgb,
                      (in_points[0], in_points[1]),
                      (in_points[2], in_points[3]),
                      (0, 255, 0), 1)
        
        if not has_matches:
            # Add text if no matches were found
            cv.putText(self.img_rgb, "No matching regions found", (50, 50), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                      
        # Save the result image
        cv.imwrite(self.output_path, self.img_rgb)
        return self.output_path


