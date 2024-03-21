import numpy as np
from PIL import Image
from skimage.transform import resize
import cv2





def preprocess_image_for_digit_recognition(uploaded_image):
    # Convert PIL Image to a NumPy array
    image_array = np.array(uploaded_image)
    
    # Convert to grayscale
    if len(image_array.shape) == 3:  # Check if the image is colored (3 channels)
        image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        image = image_array  # The image is already in grayscale

    # Thresholding the image to get a binary image
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours and the bounding rectangle
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Crop and resize the image around the digit
        digit = binary_image[y:y+h, x:x+w]
        resized_digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
        
        # Create a 28x28 image and place the digit in the center
        final_image = np.zeros((28, 28), dtype=np.uint8)
        final_image[4:24, 4:24] = resized_digit
    else:
        # If no contours are found, just resize the image
        final_image = cv2.resize(binary_image, (28, 28), interpolation=cv2.INTER_AREA)
    
    return final_image
