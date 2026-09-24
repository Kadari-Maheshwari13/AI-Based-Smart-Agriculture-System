from pathlib import Path
import json

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import joblib
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from PIL import Image


# ============================================================
# BASE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR)
)

CORS(app)


# ============================================================
# CROP RECOMMENDATION MODEL
# ============================================================

MODEL_PATH = MODEL_DIR / "crop_recommendation_model.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)


# ============================================================
# DISEASE DETECTION MODEL
# ============================================================

# Using the Finetuned model because it had the best
# Macro F1 and Weighted F1 among the tested models.

DISEASE_MODEL_PATH = (
    MODEL_DIR /
    "disease_mobilenetv3_finetuned.pth"
)

DISEASE_CLASS_PATH = (
    MODEL_DIR /
    "disease_classes_finetuned.json"
)


# ============================================================
# LOAD DISEASE CLASSES
# ============================================================

with open(DISEASE_CLASS_PATH, "r") as f:
    disease_classes = json.load(f)


# ============================================================
# CREATE MOBILE NET MODEL
# ============================================================

disease_model = mobilenet_v3_small(
    weights=None
)


# Replace final classification layer

disease_model.classifier[3] = nn.Linear(
    disease_model.classifier[3].in_features,
    len(disease_classes)
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

disease_model.load_state_dict(
    torch.load(
        DISEASE_MODEL_PATH,
        map_location="cpu"
    )
)

disease_model.eval()


# ============================================================
# DISEASE IMAGE TRANSFORMATION
# ============================================================

disease_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# DISEASE SUGGESTIONS
# ============================================================

disease_suggestions = {

    "Anthracnose": (
        "Remove severely affected leaves or plant parts, "
        "maintain good air circulation, avoid excessive leaf wetness, "
        "and consult a local agricultural expert if the infection spreads."
    ),

    "Aphids": (
        "Inspect the undersides of leaves, remove heavily infested parts, "
        "control ants around the plants, and monitor the plant regularly."
    ),

    "Fall_Armyworm": (
        "Inspect leaves and growing points for larvae or feeding damage, "
        "remove heavily affected parts where practical, "
        "and monitor the crop regularly."
    ),

    "Healthy": (
        "The image appears healthy. Continue regular monitoring, "
        "proper watering, nutrition, and good field hygiene."
    ),

    "Leaf_Curl_Virus": (
        "Remove severely affected plants where appropriate, "
        "control insect vectors such as whiteflies, "
        "remove weeds around the crop, "
        "and consult an agricultural expert for management advice."
    ),

    "Mites": (
        "Inspect the undersides of leaves for mites and webbing, "
        "remove severely affected leaves where practical, "
        "and monitor surrounding plants for spread."
    ),

    "Mosaic_Virus": (
        "Remove severely affected plants where appropriate, "
        "control insect vectors, maintain field hygiene, "
        "and use healthy planting material."
    ),

    "Powdery_Mildew": (
        "Improve air circulation, avoid excessive humidity around foliage, "
        "remove severely affected leaves, "
        "and monitor nearby plants."
    ),

    "Shoot_and_Fruit_Borer": (
        "Inspect shoots and fruits for entry holes or larvae, "
        "remove and dispose of severely affected plant parts, "
        "and monitor the crop regularly."
    ),

    "Thrips": (
        "Inspect young leaves and flowers for thrips, "
        "remove severely damaged plant parts where practical, "
        "control weeds around the crop, "
        "and monitor regularly."
    )
}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# CROP RECOMMENDATION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # ----------------------------------------------------
        # Read inputs
        # ----------------------------------------------------

        N = float(data["N"])
        P = float(data["P"])
        K = float(data["K"])

        temperature = float(
            data["temperature"]
        )

        humidity = float(
            data["humidity"]
        )

        ph = float(
            data["ph"]
        )

        rainfall = float(
            data["rainfall"]
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not (0 <= N <= 140):

            return jsonify({
                "error":
                "Nitrogen (N) must be between 0 and 140."
            }), 400


        if not (0 <= P <= 145):

            return jsonify({
                "error":
                "Phosphorus (P) must be between 0 and 145."
            }), 400


        if not (0 <= K <= 205):

            return jsonify({
                "error":
                "Potassium (K) must be between 0 and 205."
            }), 400


        if not (0 <= temperature <= 50):

            return jsonify({
                "error":
                "Temperature must be between 0 and 50 °C."
            }), 400


        if not (0 <= humidity <= 100):

            return jsonify({
                "error":
                "Humidity must be between 0 and 100%."
            }), 400


        if not (0 <= ph <= 14):

            return jsonify({
                "error":
                "pH must be between 0 and 14."
            }), 400


        if not (0 <= rainfall <= 500):

            return jsonify({
                "error":
                "Rainfall must be between 0 and 500 mm."
            }), 400


        # ----------------------------------------------------
        # Prepare input
        # ----------------------------------------------------

        input_data = np.array([[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]])


        # ----------------------------------------------------
        # Predict crop
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )

        recommended_crop = prediction[0]


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "recommended_crop":
            str(recommended_crop)

        })


    except KeyError as e:

        return jsonify({
            "error":
            f"Missing input field: {str(e)}"
        }), 400


    except ValueError:

        return jsonify({
            "error":
            "Please enter valid numeric values."
        }), 400


    except Exception as e:

        return jsonify({
            "error":
            str(e)
        }), 500


# ============================================================
# PLANT DISEASE DETECTION
# ============================================================

@app.route(
    "/predict-disease",
    methods=["POST"]
)
def predict_disease():

    try:

        # ----------------------------------------------------
        # Check uploaded image
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "error":
                "No image uploaded."
            }), 400


        image_file = request.files["image"]


        if image_file.filename == "":

            return jsonify({
                "error":
                "Please select an image."
            }), 400


        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(
            image_file
        ).convert("RGB")


        # ----------------------------------------------------
        # Transform image
        # ----------------------------------------------------

        image_tensor = disease_transform(
            image
        )


        # Add batch dimension

        image_tensor = image_tensor.unsqueeze(0)


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = disease_model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )


        # ----------------------------------------------------
        # Get prediction
        # ----------------------------------------------------

        predicted_index = (
            predicted_index.item()
        )

        confidence = (
            confidence.item()
        )


        predicted_disease = (
            disease_classes[predicted_index]
        )


        # ----------------------------------------------------
        # Get suggestion
        # ----------------------------------------------------

        suggestion = disease_suggestions.get(
            predicted_disease,

            "Monitor the plant carefully and "
            "consult a local agricultural expert."
        )


        # ----------------------------------------------------
        # Confidence warning
        # ----------------------------------------------------

        if confidence < 0.50:

            warning = (
                "The model confidence is low. "
                "Try uploading a clear image of the "
                "affected leaf or plant."
            )

        else:

            warning = ""


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "disease":
            predicted_disease,

            "confidence":
            round(
                confidence * 100,
                2
            ),

            "suggestion":
            suggestion,

            "warning":
            warning

        })


    except Exception as e:

        return jsonify({
            "error":
            str(e)
        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )