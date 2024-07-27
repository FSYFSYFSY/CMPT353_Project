import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from pykalman import KalmanFilter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.fft import fft, ifft

### Kalman Filtering
def kalmanSmooth(coef, data, outfile):
    X_valid, y_valid = data[X_columns], data[y_column]

    transition_stddev = 1.0
    observation_stddev = 0.2

    dims = X_valid.shape[-1]
    initial = X_valid.iloc[0]
    observation_covariance = np.diag([observation_stddev] * dims) ** 2
    transition_covariance = np.diag([transition_stddev] * dims) ** 2
    transition = np.identity(dims)

    transition[0, :] = coef

    kf = KalmanFilter(
        initial_state_mean=initial,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition,
    )

    kalman_smoothed, _ = kf.smooth(X_valid)

    # Fit a linear regression line to the Kalman smoothed data
    time = data['time'].values.reshape(-1, 1)
    kalman_values = kalman_smoothed[:, 0]

    regression_model = LinearRegression()
    regression_model.fit(time, kalman_values)
    regression_line = regression_model.predict(time)

    plt.figure(figsize=(15, 6))
    plt.plot(data['time'], data['speed'], 'b.', alpha=0.5, label='Observed speed')
    plt.plot(data['time'], kalman_smoothed[:, 0], 'g.', label='Kalman smoothed speed')  # Use scatter for Kalman points
    plt.plot(data['time'], regression_line, 'r-', label='Linear regression line')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.legend()
    plt.savefig(outfile)
    plt.close()

def plot_errors(model, X_valid, y_valid):
    predictions = model.predict(X_valid)
    errors = predictions - y_valid

    plt.figure(figsize=(15, 6))
    plt.plot(y_valid.index, errors, 'r.')
    plt.xlabel('Index')
    plt.ylabel('Error')
    plt.title('Prediction Errors')
    plt.show()

# Ready columns
X_columns = ['ax','ay','az','speed']
y_column = 'next_speed'

# Read file and make new column
data = pd.read_csv('Run1.csv')
data[y_column] = data['speed'].shift(-1)
data = data.dropna()

# Split data into train and valid sets
train, valid = train_test_split(data, test_size=0.3, random_state=42)
X_train, y_train = train[X_columns], train[y_column]
X_valid, y_valid = valid[X_columns], valid[y_column]

# Train KNeighborsRegressor model
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

print(f"Training score: {model.score(X_train, y_train)}")
print(f"Validation score: {model.score(X_valid, y_valid)}")

#plot_errors(model, X_valid, y_valid)

coefficients = model.predict(X_train[:1]).reshape(-1)

kalmanSmooth(coefficients, train, 'train.png')
kalmanSmooth(coefficients, valid, 'valid.png')
