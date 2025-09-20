# classifier/views.py

import numpy as np
from django.shortcuts import render
from .forms import ImageUploadForm
from .models import ImageUpload

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def classify_image(image_path):
    """
    Loads an image, preprocesses it, and uses the pre-trained model to
    return the top 3 predicted classes and their confidence scores.
    """
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)
    
    return decoded_predictions

def upload_and_recognize(request):
    """
    Handles image upload via form and displays the classification result.
    """
    # Initialize variables for both GET and POST scenarios
    form = ImageUploadForm()
    uploaded_image = None
    result = None
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            image_path = uploaded_image.image.path
            
            # Run the classification
            prediction = classify_image(image_path)
            
            # Store the result (optional, but good for history)
            # Note: Storing a list in a CharField requires a string conversion
            uploaded_image.classified_result = str(prediction)
            uploaded_image.save()
            
            # The 'result' variable for the template is the actual list of predictions
            result = prediction
            
            # Render the page with the result
            return render(request, "classify_app/home.html", {
                'form': form,
                'uploaded_image': uploaded_image,
                'result': result,
            })
        else:
            # Re-render with form errors if validation fails
            return render(request, "classify_app/home.html", {
                'form': form,
                'uploaded_image': uploaded_image,
                'result': result,
            })
            
    # This renders the initial page for a GET request
    return render(request, "classify_app/home.html", {
        'form': form,
        'uploaded_image': uploaded_image,
        'result': result,
    })
