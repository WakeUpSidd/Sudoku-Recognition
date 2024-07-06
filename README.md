# Sudoku Recognition and Redirection

This repository contains scripts for recognizing Sudoku puzzles from a live webcam feed and redirecting to specific URLs based on the recognized puzzle. The system leverages feature matching techniques using ORB and FLANN for accurate identification.

## Description

This project includes two main scripts:
1. `multiple_sudoku.py`: Recognizes and redirects based on multiple Sudoku puzzles.
2. `sudoku_scanner.py`: Recognizes and redirects based on a single Sudoku puzzle.

## Features

- **ORB Feature Detection**: Utilizes ORB (Oriented FAST and Rotated BRIEF) for feature detection and description.
- **FLANN Matching**: Implements FLANN (Fast Library for Approximate Nearest Neighbors) for efficient matching of features.
- **Homography and Inliers**: Ensures geometric consistency using homography and inlier counting.
- **Histogram Comparison**: Adds an extra layer of validation through histogram comparison.
- **Immediate URL Redirection**: Redirects to the specific URL upon successful recognition.

## Requirements

- Python 3.x
- OpenCV
- Numpy
- Webbrowser
- Tesseract (for OCR)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sudoku-recognition.git
    ```
2. Navigate to the directory:
    ```bash
    cd sudoku-recognition
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Multiple Sudoku Recognition

1. Place your reference Sudoku images in the `images` directory.
2. Update the `reference_image_paths` and `urls` lists in `multiple_sudoku.py` with the paths to your reference images and corresponding URLs.
3. Run the script:
    ```bash
    python multiple_sudoku.py
    ```

### Single Sudoku Recognition

1. Place your reference Sudoku image in the `images` directory.
2. Update the `reference_image_path` in `sudoku_scanner.py` with the path to your reference image.
3. Run the script:
    ```bash
    python sudoku_scanner.py
    ```
