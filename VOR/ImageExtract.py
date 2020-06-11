
import cv2


def extract_elements(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # grayscale
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=13)
    # dilate
    _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # get contours
    # for each contour found, draw a rectangle around it on original image

    bboxes = []
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append([x, y, w, h])
    return bboxes
