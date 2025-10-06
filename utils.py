# utils.py
import math
import numpy as np
import cv2

def calculate_angle(a, b, c):
    """Calculates the angle between three 2D points (landmarks)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def visualize_angle(image, a, b, c, angle, color=(255, 255, 255)):
    """Draws the angle and connecting lines on the image."""
    h, w, _ = image.shape
    
    a_pix = (int(a[0] * w), int(a[1] * h))
    b_pix = (int(b[0] * w), int(b[1] * h))
    c_pix = (int(c[0] * w), int(c[1] * h))

    cv2.line(image, b_pix, a_pix, color, 2)
    cv2.line(image, b_pix, c_pix, color, 2)
    cv2.circle(image, a_pix, 10, (0, 0, 255), -1)
    cv2.circle(image, b_pix, 12, (255, 0, 0), -1)
    cv2.circle(image, c_pix, 10, (0, 0, 255), -1)

    cv2.putText(image, f"{int(angle)}", 
                (b_pix[0] - 40, b_pix[1] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)