import streamlit as st
import cv2
import numpy as np
from PIL import Image

def process_image(image, threshold_val, blur_kernel, saturation_factor, canny_lower, canny_upper, min_contour_area):
    """
    Process the image by adjusting saturation, converting to grayscale,
    blurring, thresholding, detecting edges, and finding/drawing contours.
    """
    # If the image is in color (3 channels), adjust saturation.
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Multiply the saturation channel by the provided factor and clip to [0, 255]
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation_factor, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Convert to grayscale for further processing.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur; ensure kernel size is greater than 1.
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    
    # Apply a fixed threshold to get a binary image.
    ret, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Use Canny edge detection to find edges.
    edges = cv2.Canny(thresh, canny_lower, canny_upper)
    
    # Find contours from the detected edges.
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on a copy of the original image.
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    
    # Draw rectangles around large clusters of contours.
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return thresh, edges, output

def main():
    st.title("Thermal Image Object Detection")
    st.write("Upload a thermal image and adjust parameters to detect objects in real time.")
    
    # Sidebar for parameter adjustments.
    st.sidebar.header("Processing Parameters")
    threshold_val = st.sidebar.slider("Threshold Value", 0, 255, 127)
    blur_kernel = st.sidebar.slider("Gaussian Blur Kernel Size (odd numbers only)", 1, 21, 5, step=2)
    saturation_factor = st.sidebar.slider("Saturation Factor", 0.0, 3.0, 1.0, step=0.1)
    canny_lower = st.sidebar.slider("Canny Edge Lower Threshold", 0, 255, 50)
    canny_upper = st.sidebar.slider("Canny Edge Upper Threshold", 0, 255, 150)
    min_contour_area = st.sidebar.slider("Minimum Contour Area", 0, 5000, 100)
    
    # Image uploader widget.
    uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Process the image using the specified parameters.
        thresh, edges, output = process_image(image, threshold_val, blur_kernel, saturation_factor, canny_lower, canny_upper, min_contour_area)
        
        st.subheader("Thresholded Image")
        st.image(thresh, clamp=True, channels="GRAY")
        
        st.subheader("Canny Edges")
        st.image(edges, clamp=True, channels="GRAY")
        
        st.subheader("Detected Contours")
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), channels="RGB")

if __name__ == "__main__":
    main()