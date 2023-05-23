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
    # Convert the image to RGB mode and discard alpha channel if present
    img = image.convert("RGB")

    # Resize the image according to the model's requirements
    img = img.resize((299, 299))
    
    # Convert to numpy array and preprocess
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Perform object detection
    features = model.predict(x)
    
    # Return the top predicted labels with confidence scores
    labels = decode_predictions(features, top=K)[0]
    
    return labels


# Main Streamlit app
def main():
    image = Image.open('invoke_logo.jpg')
    st.image(image, caption="Version 230523", use_column_width=True)
    st.title("Object Detection App")

    # Create a file uploader for image files
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['png', 'jpeg'])

    if uploaded_files:
        # Create an empty list to store the results
        results = []

        # Iterate over the uploaded files
        for file in uploaded_files:
            # Read the image file
            img = Image.open(file)

            # Perform object detection
            labels = perform_object_detection(img)

            # Extract the file name, labels, and confidence scores
            file_name = file.name
            label_names = [label[1] for label in labels]
            confidence_scores = [label[2] for label in labels]

            # Append the results to the list
            results.append({"File Name": file_name, "Label": label_names, "Confidence Score": confidence_scores})

        # Create the dataframe from the results list
        results_df = pd.DataFrame(results)

        # Show the results as a CSV file
        st.subheader("Results")
        st.dataframe(results_df)

        # Generate and download the CSV file
        csv_data = results_df.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="object_detection_results.csv")


# Run the app
if __name__ == "__main__":
    main()
