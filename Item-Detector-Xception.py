import streamlit as st
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
from PIL import Image

# Load the Xception model
model = Xception(weights='imagenet', include_top=True)

# Define the number of objects to detect
K = 5

# Function to perform object detection on an image
def perform_object_detection(image):
    # Resize the image according to the model's requirements
    img = image.resize((299, 299))
    
    # Convert to numpy array and preprocess
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Perform object detection
    features = model.predict(x)
    
    # Return the top predicted labels
    labels = decode_predictions(features, top=K)[0]
    
    return labels

# Main Streamlit app
def main():
    st.title("Object Detection App")
    
    # Create a file uploader for image files
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)
    
    if uploaded_files:
        # Create an empty dataframe to store the results
        results_df = pd.DataFrame(columns=["File Name", "Label"])
        
        # Iterate over the uploaded files
        for file in uploaded_files:
            # Read the image file
            image = Image.open(file)
            
            # Perform object detection
            labels = perform_object_detection(image)
            
            # Extract the file name and labels
            file_name = file.name
            label_names = [label[1] for label in labels]
            
            # Append the results to the dataframe
            results_df = results_df.append({"File Name": file_name, "Label": label_names}, ignore_index=True)
        
        # Show the results as a CSV file
        st.subheader("Results")
        st.dataframe(results_df)
        
        # Generate and download the CSV file
        csv_data = results_df.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="object_detection_results.csv")

# Run the app
if __name__ == "__main__":
    main()
