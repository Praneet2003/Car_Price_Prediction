
import time
import os
import json
import threading

from flask import Flask, request, jsonify, send_from_directory

from EV_Range_Prediction.model import train_model, predict_range

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
OPTIONS_PATH = os.path.join(BASE_DIR, "options.json")


# -----------------------------------
# Load Cached Dropdown Options
# -----------------------------------

with open(OPTIONS_PATH, "r", encoding="utf-8") as file:
    options = json.load(file)


# -----------------------------------
# Model Initialization
# -----------------------------------

model_objects = None
model_lock = threading.Lock()


def get_model():
    global model_objects

    if model_objects is None:
        with model_lock:
            if model_objects is None:
                print("Training model...")
                model_objects = train_model()

    return model_objects


# -----------------------------------
# Home Route
# -----------------------------------

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/frontend/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# -----------------------------------
# Get Vehicle Options
# -----------------------------------

@app.route("/api/options", methods=["GET"])
def get_options():

    return jsonify(options)


# -----------------------------------
# Prediction Route
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid request data"}), 400

        model_year = int(data["model_year"])
        make = data["make"]
        model = data["model"]
        vehicle_type = data["vehicle_type"]

        # Input Validation

        if model_year < options["min_year"] or model_year > options["max_year"]:
            return jsonify({
                "error": f"Model year must be between {options['min_year']} and {options['max_year']}"
            }), 400

        if make not in options["make_models"]:
            return jsonify({"error": "Invalid vehicle manufacturer"}), 400

        if model not in options["make_models"][make]:
            return jsonify({
                "error": "Invalid vehicle model for the selected manufacturer"
            }), 400

        if vehicle_type not in options["vehicle_types"]:
            return jsonify({"error": "Invalid vehicle type"}), 400

        # -----------------------------------
        # Initialize Model + Measure Time
        # -----------------------------------

        start_time = time.time()

        model_objects = get_model()

        print(
            "Model loading time:",
            round(time.time() - start_time, 2),
            "seconds"
        )

        (
            spark,
            pipeline_model,
            assembler,
            dt_model,
            make_models,
            makes,
            vehicle_types,
            min_year,
            max_year
        ) = model_objects

        # -----------------------------------
        # Generate Prediction + Measure Time
        # -----------------------------------

        prediction_start = time.time()

        predicted_range = predict_range(
            spark,
            pipeline_model,
            assembler,
            dt_model,
            model_year,
            make,
            model,
            vehicle_type
        )

        print(
            "Prediction time:",
            round(time.time() - prediction_start, 2),
            "seconds"
        )

        # -----------------------------------
        # Convert Miles to Kilometres
        # -----------------------------------

        predicted_range_km = predicted_range * 1.60934

        # -----------------------------------
        # Return Prediction
        # -----------------------------------

        return jsonify({
            "predicted_range_miles": round(predicted_range, 2),
            "predicted_range_km": round(predicted_range_km, 2)
        })

    except (KeyError, ValueError, TypeError):

        return jsonify({
            "error": "Please provide valid vehicle details"
        }), 400

    except Exception:

        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed. Please try again."
        }), 500


# -----------------------------------
# Run Flask Application
# -----------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
