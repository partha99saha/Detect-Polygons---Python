import cv2
import numpy as np
import random


def get_shape_name(cnt, approx):
    sides = len(approx)

    if sides == 3:
        return "Triangle"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif sides == 5:
        return "Pentagon"
    elif sides == 6:
        return "Hexagon"
    else:
        # Check if circle using area vs enclosing circle
        area = cv2.contourArea(cnt)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius**2)
        if 0.8 <= area / circle_area <= 1.2:
            return "Circle"
        elif not cv2.isContourConvex(approx):
            return "Star"
        elif len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, orientation) = ellipse
            major_axis, minor_axis = max(axes), min(axes)
            ratio = minor_axis / major_axis
            return "Circle" if ratio > 0.85 else "Oval"
        else:
            return "Polygon"


def detect_polygons(image_path, min_area=100):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold for better results in different lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        shape_name = get_shape_name(cnt, approx)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.drawContours(img, [approx], -1, color, 3)
        x, y = approx[0][0]
        cv2.putText(
            img, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    cv2.imshow("Detected Shapes", img)
    cv2.imwrite(f"detected_shapes_{image_path}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect_polygons("geometric-shapes.jpg")
