from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model using pickle
with open('best_model_weights.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    attributes = ['season', 'year', 'month', 'holiday','hour', 'weekday', 'working_day', 'weathers_it',
                  'temp', 'atemp', 'humidity', 'windspeed']
    
    input_values = [float(request.form[attr]) for attr in attributes]
    input_array = np.array(input_values).reshape(1, -1)
    
    prediction = model.predict(input_array)
    
    return render_template('index.html', prediction=f'Predicted count: {prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
