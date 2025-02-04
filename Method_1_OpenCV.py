import numpy as np
import cv2
from pyzbar.pyzbar import decode

def detect(image):
    if image is None:
        print("Error: Image is empty or not loaded correctly.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)  # Show grayscale image

    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    cv2.imshow("Gradient", gradient)  # Show gradient image

    blurred = cv2.GaussianBlur(gradient, (5, 5), 0)
    cv2.imshow("Blurred Image", blurred)  # Show blurred image

    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresholded Image", thresh)  # Show thresholded image

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Morphological Closing", closed)  # Show morphological closing

    closed = cv2.erode(closed, kernel, iterations=3)
    cv2.imshow("Erosion", closed)  # Show after erosion

    closed = cv2.dilate(closed, kernel, iterations=3)
    cv2.imshow("Dilation", closed)  # Show after dilation

    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h


def scanner(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (800, 800))
    bbox = detect(image)
    if bbox is not None:
        x, y, w, h = bbox
        roi = image[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        decoded_objects = decode(roi)
        if decoded_objects:
            print("Barcode detected:")
            for obj in decoded_objects:
                print("Type:", obj.type)
                print("Data:", obj.data.decode("utf-8"))
        else:
            print("No barcode detected.")

        cv2.imshow("Detected Barcode", image)
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
image_path = 'images/example_barcode5.png'
scanner(image_path)
