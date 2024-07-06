# Testing out the recognition and redirection on multiple sudoku images.

import cv2
import numpy as np
import webbrowser
import time

# Paths to the reference Sudoku images
reference_image_paths = [
    r'C:\sudoku-recognition\images\printable_sudoku_20240617_070630.png',
    r'C:\sudoku-recognition\images\printable_sudoku_20240617_070743.png'
]

# URLs to open for each Sudoku image
urls = [
    "https://drive.google.com/file/d/1-WEKxAPhMbbuwipQ5bTlYt3pcyjqUag0/view?usp=sharing",
    "https://drive.google.com/file/d/1-XEP44rgrnKjRiQTYU2dBjSC5wpUaazW/view?usp=sharing"
]

# Load the reference Sudoku images
reference_images = [cv2.imread(path, 0) for path in reference_image_paths]

# Check if reference images are loaded correctly
for i, img in enumerate(reference_images):
    if img is None:
        print(f"Error: Reference image {reference_image_paths[i]} could not be loaded.")
        exit()
    else:
        print(f"Reference image {i} loaded successfully.")

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Compute keypoints and descriptors for the reference images
ref_keypoints, ref_descriptors = zip(*[orb.detectAndCompute(img, None) for img in reference_images])

# Initialize FLANN matcher
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def get_good_matches(matches):
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    return good_matches

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute keypoints and descriptors for the frame
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
    if des_frame is None:
        print("No descriptors found in frame.")
        continue

    match_found = False
    highest_inliers = 0
    best_match_index = -1

    for i, (kp_ref, des_ref) in enumerate(zip(ref_keypoints, ref_descriptors)):
        matches = flann.knnMatch(des_ref, des_frame, k=2)
        good_matches = get_good_matches(matches)

        print(f"Number of good matches for image {i}: {len(good_matches)}")

        if len(good_matches) > 30:  # Initial threshold for good matches
            # Apply homography to check for geometric consistency
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is None:
                print(f"Homography could not be computed for image {i}.")
                continue

            matches_mask = mask.ravel().tolist()

            inliers = np.sum(matches_mask)
            print(f"Number of inliers for image {i}: {inliers}")

            if inliers > highest_inliers:
                highest_inliers = inliers
                best_match_index = i

    if highest_inliers > 30:  # Threshold for inliers to open a URL
        print(f"Match found for image {best_match_index} with {highest_inliers} inliers! Opening the URL...")
        webbrowser.open(urls[best_match_index])
        break
    else:
        print("No sufficient match found. Ignoring this frame.")

    time.sleep(2)  # Wait for 2 seconds before capturing a frame again

cap.release()
cv2.destroyAllWindows()
