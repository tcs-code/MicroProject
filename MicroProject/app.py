from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load pre-trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Load and preprocess the data
team_data = pd.read_csv('team_data.csv')
scaler = StandardScaler()
features = team_data.drop(columns=['Team'])
scaler.fit(features)

teams = ['CSK', 'RCB', 'MI', 'PBKS', 'RR', 'DC', 'KKR', 'SRH']

def preprocess_input(team1, team2):
    team1_data = team_data[team_data['Team'] == team1].drop(columns=['Team']).values.flatten()
    team2_data = team_data[team_data['Team'] == team2].drop(columns=['Team']).values.flatten()
    combined_data = np.concatenate((team1_data, team2_data))
    combined_data = scaler.transform(combined_data.reshape(1, -1))
    return combined_data.reshape(1, combined_data.shape[1], 1)

@app.route('/')
def index():
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    team1 = data['team1']
    team2 = data['team2']
    input_data = preprocess_input(team1, team2)
    win_probabilities = model.predict(input_data)
    total_prob = np.sum(win_probabilities, axis=1, keepdims=True)
    team1_win_prob = (win_probabilities[:, 0] / total_prob[:, 0]) * 100
    team2_win_prob = (win_probabilities[:, 1] / total_prob[:, 0]) * 100
    return jsonify({
        'team1': team1,
        'team2': team2,
        'team1_win_prob': team1_win_prob[0],
        'team2_win_prob': team2_win_prob[0]
    })

if __name__ == '__main__':
    app.run(debug=True)
