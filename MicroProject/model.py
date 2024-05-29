import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Step 1: Data Loading and Preprocessing
# Load the historical cricket match data
cricket_data = pd.read_csv('/content/cricket_data_with_prediction_split_and_RR.csv')

# Step 2: Feature Engineering
# Extract relevant features and target variables
X = cricket_data.drop(columns=['Team1_name', 'Team2_name', 'Team1_win%', 'Team2_win%'])
y = cricket_data[['Team1_win%', 'Team2_win%']]

# Normalize the features
X = (X - X.mean()) / X.std()

# Reshape the input data to be suitable for CNN: (samples, time steps, features)
X = X.values.reshape((X.shape[0], X.shape[1], 1))
y = y.values

# Step 3: Model Training
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X.shape[1], 1)),
    Dropout(0.5),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(2)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mse', 'mae'])
model.summary()

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Step 4: Model Evaluation
# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Step 5: Prediction
# Prompt the user to enter data for both teams
example_data = []
for i in range(2):
    print(f"Enter data for Team {i+1}:")
    data = []
    data.append(float(input(f"Enter Matches played by Team {i+1}: ")))
    data.append(float(input(f"Enter Matches won by Team {i+1}: ")))
    for j in range(11):
        sr = float(input(f"Enter Batsman {j+1} Strike Rate: "))
        data.append(sr)
    for j in range(5):
        eco = float(input(f"Enter Bowler {j+1} Economy: "))
        data.append(eco)
    rr = float(input(f"Enter Team {i+1} Run Rate: "))
    data.append(rr)
    example_data.extend(data)

# Normalize the input data
example_data = np.array(example_data)
example_data = (example_data - cricket_data.mean()) / cricket_data.std()

example_data = example_data.reshape(1, -1, 1)

# Predict win probabilities for both teams
win_probabilities = model.predict(example_data)

# Normalize probabilities to sum up to 100%
total_prob = np.sum(win_probabilities, axis=1, keepdims=True)
team1_win_prob = (win_probabilities[:, 0] / total_prob[:, 0]) * 100
team2_win_prob = (win_probabilities[:, 1] / total_prob[:, 0]) * 100

# Print results
print(f"Team 1 has a {team1_win_prob[0]:.2f}% chance of winning")
print(f"Team 2 has a {team2_win_prob[0]:.2f}% chance of winning")
