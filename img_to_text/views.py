from rest_framework.decorators import api_view
from django.http import JsonResponse
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import numpy as np
from .utils import image_preprocessing, generate_predictions, postprocessingv1, \
    postprocessingv2, extract_text_from_prediction, load_model

# Load the YOLO model once (global variable)
model = load_model(model_choice="best.pt")

csrf_exempt()
def detect_text(request):
     if request.method == 'GET':
           return render(request, 'home.html')
     
     if request.method == 'POST':
        try:
            # Get image from the POST request
            image_file = request.FILES.get('image')

            image_ = image_preprocessing(image_file)
            # print(np.array(image).astype(np.uint8).max())

            image = np.array(image_).astype(np.uint8)

            # Generate predictions from model
            predictions = generate_predictions(image, model)

            # Postprocess predictions
            results, _ = postprocessingv1(predictions)
            print("Predictions:", predictions)

            results, detections = postprocessingv2(predictions, results)

            extracted_text = extract_text_from_prediction(predictions, centroid_ranking = True)

            prefix = f"Patient Diagnosis:\n\n"

            if detections is None:
                string = "No Stenosis Detected! Please check and confirm."

            else:
                class_detections = list(set(detections))
                if len(class_detections) == 1:
                    message = class_detections[0] + " Detected!"
                elif len(class_detections) == 2:
                    message = " and ".join([t.replace(" Stenosis", "") for t in class_detections])
                    message = message + " Stenosis Detected!"
                else:
                    message = ", ".join([t.replace(" Stenosis", "") for t in class_detections[:-1]])
                    message = message + ", and " + class_detections[-1] + " Detected!"

                string = message


            return JsonResponse({"Extracted_text": string})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
