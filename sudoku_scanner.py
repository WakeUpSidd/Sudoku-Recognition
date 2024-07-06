# Testing out the recognition and redirection on a single sudoku image.

import cv2
import numpy as np
import webbrowser
import os
import time

# Correctly format the file path
reference_image_path = r'C:\sudoku-recognition\images\printable_sudoku_20240617_070515.png'

# Load the reference Sudoku image
reference_image = cv2.imread(reference_image_path, 0)
if reference_image is None:
    print("Error: Reference image not found.")
    exit()

# Apply Gaussian blur and adaptive thresholding to the reference image to enhance edges
reference_image = cv2.GaussianBlur(reference_image, (5, 5), 0)
reference_image = cv2.adaptiveThreshold(reference_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create SIFT detector
sift = cv2.SIFT_create()

# Compute keypoints and descriptors for the reference image
keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_image, None)

# Define FLANN based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

detection_threshold = 30  # Minimum number of good matches to consider it a detection
consecutive_detections = 0  # Counter for consecutive frames with enough good matches
required_consecutive_detections = 2  # Number of consecutive frames required

frame_counter = 0
output_dir = "captured_frames"
os.makedirs(output_dir, exist_ok=True)

def compare_histograms(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Check if the frame is valid
    if frame is None or frame.size == 0:
        print("Error: Captured frame is empty.")
        continue

    # Save the frame to disk for debugging
    frame_path = os.path.join(output_dir, f"frame_{frame_counter}.png")
    cv2.imwrite(frame_path, frame)
    frame_counter += 1

    # Resize the frame for better detection at a distance
    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur and adaptive thresholding to the frame
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    gray_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Compute keypoints and descriptors for the frame
    keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)

    # Check if descriptors are found in the frame
    if descriptors_frame is not None and len(descriptors_frame) >= 2:
        # Match descriptors using FLANN matcher
        matches = flann.knnMatch(descriptors_ref, descriptors_frame, k=2)

        # Store all good matches as per Lowe's ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        # Debug information
        print(f"Number of good matches: {len(good_matches)}")

        # If enough matches are found, increase the counter
        if len(good_matches) >= detection_threshold:
            # Check homography to verify geometric consistency
            src_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            if sum(matchesMask) > detection_threshold * 0.7:  # Check if enough inliers are found
                # Compare histograms for an additional check
                score = compare_histograms(reference_image, gray_frame)
                print(f"Histogram similarity score: {score}")
                if score > 0.9:  # Set a suitable threshold for histogram similarity
                    consecutive_detections += 1
                    print(f"Consecutive detections: {consecutive_detections}")
                else:
                    consecutive_detections = 0
                    print("Resetting consecutive detections due to insufficient histogram similarity.")
            else:
                consecutive_detections = 0
                print("Resetting consecutive detections due to insufficient geometric consistency.")
        else:
            consecutive_detections = 0
            print("Resetting consecutive detections due to insufficient good matches.")

        # If we have enough consecutive detections, open the URL
        if consecutive_detections >= required_consecutive_detections:
            print("Match found! Opening the URL...")
            webbrowser.open("https://youtu.be/dQw4w9WgXcQ?si=kCgnunaa8udtWB32")
            break
    else:
        print("Not enough descriptors found in frame.")
        consecutive_detections = 0

    # Wait briefly before capturing the next frame
    time.sleep(0.2)

# Release the capture
cap.release()
cv2.destroyAllWindows()
