# Import libraries

import numpy as np
import cv2

from PIL import Image

from ultralytics import YOLO
import warnings

__all__ = [
    "extract_bbox",
    "extracted_text",
    "extract_text_from_prediction",
    "generate_image_with_bounding_box",
    "generate_predictions",  # STEP 3: Make predictions on image using model
    "image_preprocessing", # STEP 2: Load and process image
    "filter_predictions_v1",
    "filter_predictions",
    "postprocessingv1",
    "postprocessingv2", # STEP 4: Process the model predictions
    "load_model" # STEP 1: Load model
]

# Ignore all non-failing warnings
warnings.filterwarnings("ignore")

# Set visualization colors
# COLORS = [
#     (0, 255, 250),  # Grade 0 Stenosis
#     (255, 200, 0),  # Grade 1 Stenosis
#     (0, 40, 255),   # Grade 2 Stenosis
#     (200, 40, 255)  # Grade 3 Stenosis
# ]

COLORS = [
    (0, 255, 250) for _ in range(60)
]

MODEL_MAP = {
    "YOLO V8 (Small)": "yolov8s",
    "YOLO V8 (Small) (Segment)": "yolov8s-seg",
    # "YOLO V8 (Medium)": "yolov8m",
    # "YOLO V8 (Large)": "yolov8l",
    # "YOLO V9 (Small)": "yolov9s",
    "YOLO V9 (Medium)": "yolov9c",
    # "YOLO V9 (Large)": "yolov9l",
}

MODEL_CHOICES = list(MODEL_MAP.keys())

def load_model(model_choice = "yolov8"):
    """Load trained stenosis detection model"""

    return YOLO(f'yolo/{model_choice}')

def image_preprocessingv1(uploaded_file):
    """Preprocess uploaded StreamLit file."""

    loaded_image = np.array(Image.open(uploaded_file), dtype="uint8")

    _, loaded_image = cv2.threshold(
        loaded_image.astype(np.uint8),
        127, 255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)

    loaded_image = cv2.dilate(loaded_image, kernel, iterations=1)
    # loaded_image = cv2.erode(loaded_image, kernel, iterations=1)

    return loaded_image

def image_preprocessing(uploaded_file):
    """Preprocess uploaded StreamLit file."""

    loaded_image = np.array(Image.open(uploaded_file), dtype="uint8")

    loaded_image = cv2.resize(loaded_image, (640, 640))
    _, loaded_image = cv2.threshold(
        loaded_image.astype(np.uint8),
        127, 255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)

    # print(loaded_image)

    loaded_image = cv2.dilate(255 - loaded_image, kernel, iterations=2)
    loaded_image = cv2.erode(loaded_image, kernel, iterations=1)

    loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)

    _, loaded_image = cv2.threshold(
        255 - loaded_image,
        127, 255,
        cv2.THRESH_BINARY
    )

    print("Unique pixel values:", np.unique(loaded_image))
    print("Image shape:", loaded_image.shape)

    return loaded_image


def image_preprocessingv1(uploaded_file):
    image = cv2.imread(uploaded_file)

    _, image = cv2.threshold(image.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    return image


def generate_predictions(image, model, conf=0.3, iou=.8):
    """Pass an image through the trained model and return the predicted results."""
    img = np.asarray(image) if not isinstance(image, np.ndarray) else image

    if len(img.shape) < 3:
        img = np.stack([img, img, img], axis = -1)

    prediction = model.predict(
        img,
        imgsz=640,
        conf=conf,
        agnostic_nms=True,
        iou=iou,
    )
    return prediction[0]


def postprocessingv1(prediction):
    """Take the predicted output of the model and format it for further use."""
    boxes = prediction.boxes
    return (prediction.plot(), False) if boxes.shape[0] != 0 else (prediction.orig_img, True)


def extract_bbox(result):
    """Extract coordinates for displaying bounding boxes."""
    info = result.summary()

    class_names = [r['name'] for r in info]
    class_indices = [r['class'] for r in info]

    bbox_info = [r['box'] for r in info]

    print(bbox_info)

    bbox_info = [
        [(int(b['x1']), int(b['y1'])), (int(b['x2']), int(b['y2']))]
        for b in bbox_info
    ]

    centroids = [
        ((min_point[0] + max_point[0]) // 2, (min_point[1] + max_point[1]) // 2)
        for min_point, max_point in bbox_info
    ]

    return [
        {
            "name": name,
            "index": index,
            "box": box,
            "centroid": centroid,
        }
        for name, index, box, centroid in zip(class_names, class_indices, bbox_info, centroids)
    ]


def generate_image_with_bounding_box(result, image=None):
    extracted_result = extract_bbox(result)
    if image is None:
        image = result.orig_img

    class_names = []

    for bbox in extracted_result:
        print(bbox)

        print("="*20 + "\n\n")
        upper_left, lower_right = bbox['box']
        class_name = bbox['name']
        class_names.append(class_name)
        class_index = bbox['index']

        color = COLORS[class_index]

        lower_left = (int(upper_left[0]), int(lower_right[1]))
        upper_right = (int(lower_right[0]), int(upper_left[1]))

        image = cv2.line(image, upper_left, lower_left, color=color)
        image = cv2.line(image, lower_left, lower_right, color=color)
        image = cv2.line(image, lower_right, upper_right, color=color)
        image = cv2.line(image, upper_right, upper_left, color=color)

    return image, class_names


def postprocessingv2(prediction, image=None):
    """Take the predicted output of the model and format it for further use."""
    boxes = prediction.boxes
    return generate_image_with_bounding_box(prediction, image) if boxes.shape[0] != 0 else (prediction.orig_img, None)

def filter_predictions_v1(predictions):
    unique_classes = [p['name'] for p in predictions]
    double_classes = [p for p in unique_classes if unique_classes.count(p) > 1]

    if len(double_classes) == 0:
        return predictions

    suspicious_predictions = list(filter(lambda p: p['name'] in double_classes, predictions))

    for prediction in suspicious_predictions:
        centroid = prediction['centroid']
        for prediction2 in suspicious_predictions:
            centroid2 = prediction2['centroid']
            if centroid == centroid2 and prediction['name'] != prediction2['name']:
                continue

            x_distance_sq = (centroid[0] - centroid2[0]) ** 2
            y_distance_sq = (centroid[1] - centroid2[1]) ** 2

            if (prediction['name'] + prediction2['name']).isupper():
                distance_limit = 5
                centroid_distance = (x_distance_sq + y_distance_sq) ** .5
            else:
                distance_limit = 2
                centroid_distance = (x_distance_sq) ** .5

            if centroid_distance < distance_limit:
                predictions.remove(prediction2)
                suspicious_predictions.remove(prediction2)

    return predictions


def filter_predictions(predictions):
    unique_classes = [p['name'] for p in predictions]
    double_classes = set([p for p in unique_classes if unique_classes.count(p) > 1])
    double_classes = list(double_classes)

    if len(double_classes) == 0:
        return predictions

    suspicious_predictions = list(filter(lambda p: p['name'] in double_classes, predictions))

    for prediction in suspicious_predictions:
        print(f"Comparing the {prediction['name']}s ...\n")

        centroid = prediction['centroid']

        suspicious_predictions_ = list(
            filter(
                lambda p: p['name'] == prediction['name'] and p['name'] in double_classes,
                suspicious_predictions
            )
        )

        print(prediction, "\n")

        for i in suspicious_predictions_:
            print(i)

        for prediction2 in suspicious_predictions_:
            centroid2 = prediction2['centroid']

            if centroid == centroid2 and prediction['name'] != prediction2['name']:
                continue

            if prediction == prediction2:
                continue

            x_distance_sq = (centroid[0] - centroid2[0]) ** 2
            y_distance_sq = (centroid[1] - centroid2[1]) ** 2

            if (prediction['name'] + prediction2['name']).isupper():
                # if prediction['name'] == prediction2['name']:
                #     distance_limit = 2
                #     centroid_distance = (x_distance_sq) ** .5
                # else:
                    distance_limit = 17
                    centroid_distance = (x_distance_sq + y_distance_sq) ** .5
            else:
                distance_limit = 4
                centroid_distance = (x_distance_sq) ** .5

            if centroid_distance < distance_limit:
                predictions.remove(prediction2)
                suspicious_predictions.remove(prediction2)
    print(f"Double classes: {', '.join(double_classes)}")
    return predictions


def extract_text_from_prediction(prediction, centroid_ranking = False):
    extracted_summary = extract_bbox(prediction)
    print("Extracted summary:", extracted_summary)

    extracted_summary = filter_predictions(extracted_summary)

    if centroid_ranking:
        unique_ys = set(instance['centroid'][-1] for instance in extracted_summary)
    else:
        unique_ys = set(instance['box'][0][-1] for instance in extracted_summary)

    unique_ys = sorted(list(unique_ys))
    print("Extracted ys:", unique_ys)

    extracted_string = ""

    for y in unique_ys:
        if centroid_ranking:
            instances = list(filter(lambda x: x['centroid'][-1] == y, extracted_summary))
            instances = list(sorted(instances, key=lambda x: x['centroid'][0]))
        else:
            instances = list(filter(lambda x: x['box'][0][-1] == y, extracted_summary))
            instances = list(sorted(instances, key = lambda x: x['box'][0][0]))

        for instance in instances:
            print("Instance:", instance)
            extracted_character = instance['name']
            extracted_string = extracted_string + extracted_character

    print("Extracted string:", extracted_string)

    return extracted_string


# image_ = image_preprocessing(file)
# # print(np.array(image).astype(np.uint8).max())

# image = np.array(image_).astype(np.uint8)

# # Generate predictions from model
# predictions = generate_predictions(image, model)

# # Postprocess predictions
# results, _ = postprocessingv1(predictions)
# print("Predictions:", predictions)

# results, detections = postprocessingv2(predictions, results)

# extracted_text = extract_text_from_prediction(predictions, centroid_ranking = True)


# # Display results of prediction
# # x = random.randint(98, 99) + random.randint(0, 99) * 0.01
# # st.sidebar.error("Accuracy : " + str(x) + " %")

# prefix = f"Patient Diagnosis:\n\n"

# if detections is None:
#     string = "No Stenosis Detected! Please check and confirm."

# else:
#     class_detections = list(set(detections))
#     if len(class_detections) == 1:
#         message = class_detections[0] + " Detected!"
#     elif len(class_detections) == 2:
#         message = " and ".join([t.replace(" Stenosis", "") for t in class_detections])
#         message = message + " Stenosis Detected!"
#     else:
#         message = ", ".join([t.replace(" Stenosis", "") for t in class_detections[:-1]])
#         message = message + ", and " + class_detections[-1] + " Detected!"

#     string = f"{message}\n\nPlease check and confirm."
#     st.sidebar.warning(prefix + string)

    #
    # st.markdown("## Remedy")
    # st.info(
    #     "Please take appropriate medical action!"
    # )
