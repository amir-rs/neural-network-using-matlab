from flask import Flask, render_template, request, jsonify
import matlab.engine
import numpy as np

app = Flask(__name__)

# Start MATLAB engine
eng = matlab.engine.start_matlab()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        engine_size = float(request.form['engine_size'])
        cylinders = int(request.form['cylinders'])
        city_consumption = float(request.form['city_consumption'])
        highway_consumption = float(request.form['highway_consumption'])
        combined_consumption = float(request.form['combined_consumption'])
        combined_mpg = float(request.form['combined_mpg'])

        # Create input array
        input_data = matlab.double([
            engine_size, 
            cylinders, 
            city_consumption,
            highway_consumption, 
            combined_consumption,
            combined_mpg
        ])

        # Call MATLAB function for prediction
        result = eng.predict_co2(input_data)
        
        return jsonify({
            'success': True,
            'prediction': round(result, 2)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)