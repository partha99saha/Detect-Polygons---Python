import cv2
import numpy as np
import random


def get_shape_name(cnt, approx):
    sides = len(approx)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)

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
        # 1️⃣ Star detection first: non-convex + enough vertices + low circularity
        if not cv2.isContourConvex(approx) and sides > 6 and circularity < 0.85:
            return "Star"

        # 2️⃣ Circle or Oval detection using ellipse
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, orientation) = ellipse
            major_axis, minor_axis = max(axes), min(axes)
            ratio = minor_axis / major_axis
            if ratio > 0.85:
                return "Circle"
            else:
                return "Oval"

        # 3️⃣ Fallback for irregular polygon
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

    filtered_contours = []

    # Filter out contours fully inside another contour
    for i, cnt1 in enumerate(contours):
        if cv2.contourArea(cnt1) < min_area:
            continue
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        inside = False
        for j, cnt2 in enumerate(contours):
            if i == j:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            if x1 > x2 and y1 > y2 and (x1 + w1) < (x2 + w2) and (y1 + h1) < (y2 + h2):
                inside = True
                break
        if not inside:
            filtered_contours.append(cnt1)

    for cnt in filtered_contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        shape_name = get_shape_name(cnt, approx)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.drawContours(img, [approx], -1, color, 3)

        # Calculate contour centroid for text
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(approx)
            cx, cy = x + w // 2, y + h // 2

        cv2.putText(
            img, shape_name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    cv2.imshow("Detected Shapes", img)
    cv2.imwrite(f"detected_shapes_{image_path}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# usage
detect_polygons("geometric-shapes.jpg")
