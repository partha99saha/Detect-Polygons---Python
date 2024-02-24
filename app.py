import cv2

# read the input image
img = cv2.imread('polygon.png')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply thresholding to convert the grayscale image to a binary image
ret, thresh = cv2.threshold(gray, 50, 255, 0)

# find the contours
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours detected:", len(contours))

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    (x, y) = cnt[0, 0]

    if len(approx) >= 5:
        img = cv2.drawContours(img, [approx], -1, (0, 255, 255), 3)
        cv2.putText(img, 'Polygon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

cv2.imshow("Polygon", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
