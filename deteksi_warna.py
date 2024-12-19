import cv2
import numpy as np

# Function to draw the largest contour detected in the ROI
def draw_largest_contour_in_roi(img, contours, roi, color, label):
    roi_x, roi_y, roi_w, roi_h = roi
    roi_contours = [cnt for cnt in contours if is_contour_in_roi(cnt, roi)]
    if roi_contours:
        largest_contour = max(roi_contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        if contour_area > 1000:  # Adjust area threshold as needed
            # Adjust contour points to original frame coordinates
            largest_contour = largest_contour + np.array([roi_x, roi_y])
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Function to check if a contour is within the ROI
def is_contour_in_roi(contour, roi):
    x, y, w, h = cv2.boundingRect(contour)
    roi_x, roi_y, roi_w, roi_h = roi
    return (roi_x < x + w / 2 < roi_x + roi_w) and (roi_y < y + h / 2 < roi_y + roi_h)

# Open video capture
cap = cv2.VideoCapture(0)

# Define HSV ranges for colors
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Create morphological kernel
kernel = np.ones((5, 5), np.uint8)

while True:
    # Capture frame by frame
    ret, frame = cap.read()
    if not ret:
        break

    # Define ROI in the center of the frame
    height, width, _ = frame.shape
    roi_w, roi_h = width // 2, height // 2
    roi_x, roi_y = width // 4, height // 4
    roi = (roi_x, roi_y, roi_w, roi_h)

    # Draw ROI rectangle
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 255), 2)

    # Extract the ROI from the frame
    roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Convert ROI to HSV
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # Create masks for each color
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.add(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply morphological operations to improve mask results
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

    # Apply GaussianBlur to reduce noise further
    mask_red = cv2.GaussianBlur(mask_red, (15, 15), 0)
    mask_green = cv2.GaussianBlur(mask_green, (15, 15), 0)
    mask_blue = cv2.GaussianBlur(mask_blue, (15, 15), 0)
    mask_yellow = cv2.GaussianBlur(mask_yellow, (15, 15), 0)

    # Find contours in each mask
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the largest contour in the ROI for each color
    draw_largest_contour_in_roi(frame, contours_red, roi, (0, 0, 255), 'Red')
    draw_largest_contour_in_roi(frame, contours_green, roi, (0, 255, 0), 'Green')
    draw_largest_contour_in_roi(frame, contours_blue, roi, (255, 0, 0), 'Blue')
    draw_largest_contour_in_roi(frame, contours_yellow, roi, (0, 255, 255), 'Yellow')

    # Display the result frame
    cv2.imshow('Detected Colors', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture and destroy windows
cap.release()
cv2.destroyAllWindows()
