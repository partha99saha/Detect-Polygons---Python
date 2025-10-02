# Detect Polygons in an Image using OpenCV

This project uses **OpenCV**, a popular library for computer vision, to detect common geometric shapes (triangles, squares, rectangles, pentagons, hexagons, circles, and stars) in an image.

## About Computer Vision
**Computer Vision (CV)** is a field of artificial intelligence that enables computers to interpret and understand visual information from the world, such as images and videos. It involves processing, analyzing, and making decisions based on visual data.

Common tasks in computer vision include:
- **Object Detection** – Identifying and locating objects within an image.
- **Image Classification** – Assigning a label to an entire image.
- **Segmentation** – Dividing an image into meaningful regions.
- **Feature Extraction** – Detecting key points, edges, or shapes for analysis.

In this project, computer vision techniques like 
**grayscale conversion, thresholding, contour detection, and polygon approximation** 
are used to identify and label geometric shapes in images.

## The program works by:
1. Reading an input image.
2. Converting it to grayscale.
3. Applying adaptive thresholding to create a binary image.
4. Finding contours in the image.
5. Approximating contours to polygons.
6. Detecting the shape based on the number of vertices and contour properties.
7. Drawing the detected polygons and labeling them on the image.

## Requirements
- Python 3.x  
- OpenCV (`opencv-python`)
- NumPy (`numpy`)

## Usage
```bash
pip install -r requirements.txt
python app.py
