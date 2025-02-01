import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import io

# Initialize MediaPipe Pose with improved settings
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def process_image(image):
    # Convert PIL Image to OpenCV format while preserving colors
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Create a copy for the original image display
    original_image = image_cv.copy()
    
    # Process the image and detect pose
    results = pose.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        # Create a copy of the original image for overlay
        output_image = original_image.copy()
        
        # Draw pose landmarks directly on the colored image
        mp_drawing.draw_landmarks(
            output_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0),  # Green color for points
                thickness=3,
                circle_radius=3
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255),  # White color for connections
                thickness=2
            )
        )
        
        # Convert back to RGB for display
        return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), None

def main():
    st.title("Advanced Human Pose Detection")
    st.write("Upload a colored image to detect human pose landmarks")
    
    # Add some instructions
    st.markdown("""
    ### Instructions:
    1. Upload a clear, well-lit color photo
    2. Make sure the person is clearly visible
    3. The system will automatically detect and display pose landmarks
    """)

    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Read image
            image = Image.open(uploaded_file)
            
            # Process image
            original_image, pose_image = process_image(image)
            
            if pose_image is not None:
                # Display images side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.header("Original Image")
                    st.image(original_image, use_column_width=True)
                
                with col2:
                    st.header("Pose Detection")
                    st.image(pose_image, use_column_width=True)
                
                st.success("✅ Pose detection completed successfully!")
            else:
                st.error("No pose detected in the image. Please try another image with a clearly visible person.")
                st.image(original_image, caption="Original Image", use_column_width=True)
        
        except Exception as e:
            st.error(f"An error occurred while processing the image. Please try another image.")
            st.exception(e)

if __name__ == "__main__":
    main()
