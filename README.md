# Detect Polygons in an Image using OpenCV

This project uses **OpenCV**, a popular library for computer vision, to detect polygons in an image.  

The program works by:
1. Reading an input image.
2. Converting it to grayscale.
3. Applying thresholding to create a binary image.
4. Finding contours in the image.
5. Detecting polygons based on contour approximation.
6. Drawing the detected polygons and labeling them on the image.

## Requirements
- Python 3.x  
- OpenCV (`opencv-python`)

## Usage
```bash
pip install -r requirements.txt
python app.py
