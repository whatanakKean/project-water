from flask import Flask, jsonify
from src.pipelines.Waterlevel_pipeline import WaterLevelModel


app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        pipeline = WaterLevelModel(station_code="PPB")
        prediction = pipeline.run_prediction_pipeline()
        # Prepare the response
        response = {
            "predictions": prediction.tolist()
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)