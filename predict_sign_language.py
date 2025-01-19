import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('final_saved_model.keras')

# Function to preprocess the image
def preprocess_image(image_path, target_size=(50, 50)):
    """
    Preprocesses an input image for prediction.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size to resize the image.

    Returns:
        numpy.ndarray: Preprocessed image ready for prediction.
    """
    # Load the image
    img = load_img(image_path, target_size=target_size)
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # Normalize the image (scale pixel values between 0 and 1)
    img_array = img_array / 255.0
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the class
def predict(image_path, class_labels):
    """
    Predicts the class of a given image.

    Args:
        image_path (str): Path to the input image.
        class_labels (list): List of class labels corresponding to model output.

    Returns:
        str: Predicted class label.
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    # Predict the class
    predictions = model.predict(preprocessed_image)
    # Get the index of the highest predicted probability
    predicted_class_index = np.argmax(predictions)
    # Return the corresponding class label
    return class_labels[predicted_class_index]

# Define the class labels
class_labels = ['A', 'B', 'C']  # Replace with your actual class labels

# Example usage
if __name__ == "__main__":
    image_path = 'test_image.jpg'  # Path to the test image
    predicted_class = predict(image_path, class_labels)
    print(f"The predicted class for the image is: {predicted_class}")
