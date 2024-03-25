import numpy as np
import cv2

# Define calibration parameters
lowest_temp = 0.0  # Lowest temperature value in the calibration data
highest_temp = 100.0  # Highest temperature value in the calibration data

# Load infrared image
image_path = 'hand-infrared.jpg'  # Replace 'hand-infrared.jpg' with the actual image file path
infrared_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Apply calibration formula
def convert_pixel_to_temperature(pixel_value):
    return (pixel_value / 255.0) * (highest_temp - lowest_temp) + lowest_temp


# Define the callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Retrieve the temperature value at the cursor position
        temperature = temperature_image[y, x]
        # Display the temperature value
        print(f"Temperature at cursor position ({x}, {y}): {temperature}")


# Load the temperature image (this gives RGB values)
#temperature_image = cv2.imread('hand-infrared.jpg')

# Create a window to display the image
cv2.namedWindow('Temperature Image')

# Set the mouse callback function
cv2.setMouseCallback('Temperature Image', mouse_callback)

# Convert pixel values to temperature values
temperature_image = convert_pixel_to_temperature(infrared_image)

# Display temperature image (for visualization purposes)
cv2.imshow('Temperature Image', temperature_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
