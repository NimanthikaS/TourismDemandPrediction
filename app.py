# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import math

# try to import real predictor; if missing, we'll fallback to a local stub file
try:
    from neural_prophet_predictor import make_prediction  # your real function
except Exception as e:
    make_prediction = None
    _import_error = str(e)
else:
    _import_error = None

app = Flask(__name__)
CORS(app)  # allow cross-origin requests (use restrictively in production)

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# expected input keys and types (float or int)
EXPECTED_KEYS = {
    "year": int,
    "month": int,
    "ccpi": float,
    "exchange_rate": float,
    "covid_impact": float,
    "easter_attack": float,
    "economic_crisis": float,
    "political_crisis": float
}

@app.route('/')
def index():
    # Simple index - requires templates/index.html to exist if you want a UI.
    try:
        return render_template('index.html')
    except Exception:
        return "Index template not found. Use /predict endpoint (POST JSON).", 200

def _validate_and_convert(data: dict):
    """
    Validate presence of keys and convert to the expected type.
    Returns (converted_dict, errors_list).
    """
    errors = []
    converted = {}

    for k, typ in EXPECTED_KEYS.items():
        if k not in data:
            errors.append(f"Missing required field: {k}")
            continue
        val = data[k]
        # allow strings that look like numbers
        try:
            if typ is int:
                converted[k] = int(float(val))
            elif typ is float:
                converted[k] = float(val)
            else:
                converted[k] = val
            # optional sanity checks
            if k == "month" and not (1 <= converted[k] <= 12):
                errors.append("month must be in 1..12")
        except Exception:
            errors.append(f"Field {k} must be convertible to {typ.__name__}; got '{val}'")
    return converted, errors

@app.route('/predict', methods=['POST'])
def predict():
    # parse JSON
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logger.exception("Failed to parse JSON")
        return jsonify(success=False, error="Invalid JSON body"), 400

    if not isinstance(data, dict):
        return jsonify(success=False, error="JSON body must be an object/dict"), 400

    # validate & convert
    converted, errors = _validate_and_convert(data)
    if errors:
        return jsonify(success=False, errors=errors), 400

    # if predictor import failed, return helpful message (and optionally use a stub)
    if make_prediction is None:
        msg = "Prediction function not available (neural_prophet_predictor import failed)."
        logger.warning(msg + (" Import error: " + _import_error) if _import_error else msg)
        # For development, you can return a dummy/fallback prediction or an explicit error.
        return jsonify(success=False, error=msg, import_error=_import_error), 500

    # call the predictor
    try:
        # ensure we pass only the expected args, in stable order
        prediction = make_prediction(
            year=converted['year'],
            month=converted['month'],
            ccpi=converted['ccpi'],
            exchange_rate=converted['exchange_rate'],
            covid_impact=converted['covid_impact'],
            easter_attack=converted['easter_attack'],
            economic_crisis=converted['economic_crisis'],
            political_crisis=converted['political_crisis']
        )

        # ensure numeric result
        if isinstance(prediction, (list, tuple)):
            # if predictor returns multiple quantiles etc, return them directly
            payload_prediction = prediction
            human_message = "Prediction returned as list/tuple."
        else:
            try:
                # attempt to cast to a float for formatting
                num = float(prediction)
                if math.isfinite(num):
                    payload_prediction = num
                    human_message = f"Predicted tourist arrivals: {int(round(num)):,}"
                else:
                    payload_prediction = prediction
                    human_message = "Prediction is not a finite number."
            except Exception:
                payload_prediction = prediction
                human_message = "Prediction could not be coerced to number."

        return jsonify(success=True, prediction=payload_prediction, message=human_message), 200

    except Exception as e:
        logger.exception("Error during model prediction")
        return jsonify(success=False, error=str(e)), 500


if __name__ == '__main__':
    # For development only. In production use a WSGI server.
    app.run(debug=True, host='0.0.0.0', port=5001)
