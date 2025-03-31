import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def decode_base64_image(base64_string):
    """Decodes a base64-encoded image into an OpenCV image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def encode_image_to_base64(image):
    """Encodes an OpenCV image to a base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def detect_features(image_b64):
    """Detects doors, caves, and dark areas in an image and overlays highlights."""
    image = decode_base64_image(image_b64)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to detect dark areas (caves, shadows, etc.)
    _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Define color ranges for possible doors (brown/wooden) and caves (dark openings)
    lower_brown = np.array([10, 50, 50], dtype=np.uint8)
    upper_brown = np.array([30, 255, 255], dtype=np.uint8)
    
    lower_dark = np.array([0, 0, 0], dtype=np.uint8)
    upper_dark = np.array([50, 50, 50], dtype=np.uint8)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Masks for doors and caves
    door_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    cave_mask = cv2.inRange(image, lower_dark, upper_dark)
    
    # Find contours for detected regions
    contours_dark, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_door, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_cave, _ = cv2.findContours(cave_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Overlay highlights on the original image
    highlighted_image = image.copy()
    cv2.drawContours(highlighted_image, contours_door, -1, (0, 255, 0), 2)  # Green for doors
    cv2.drawContours(highlighted_image, contours_cave, -1, (255, 0, 0), 2)  # Blue for caves
    cv2.drawContours(highlighted_image, contours_dark, -1, (0, 0, 255), 2)  # Red for dark areas
    
    # Encode the image back to base64
    highlighted_image_b64 = encode_image_to_base64(highlighted_image)
    
    return highlighted_image_b64

