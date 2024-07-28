import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pykalman import KalmanFilter

#Smooth User speed over time for entire list of dataframe
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

def lowess_smooth(df, label):
    output_dir = 'lowess_smoothed'
    
    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'])

    # Read and smooth processed data
    smoothed = sm.nonparametric.lowess(df['speed'], df['time'], frac=0.085)

    # Create DataFrame with smoothed data
    smoothed_df = pd.DataFrame({'time': df['time'], 'speed': smoothed[:, 1]})

    # Generate the output file name
    output_file = os.path.join(output_dir, f'{label}_smoothed.csv')

    # Save the smoothed data to CSV files
    smoothed_df.to_csv(output_file, index=False)

	#Output plot
	#plt.figure(figsize=(12, 4))
	#plt.plot(file['time'], file['speed'], 'b.', alpha=0.5, label='Raw Data')
	#plt.plot(run_df['time'], run_loess_smoothed[:, 1], label='LOWESS', color='red')
	#plt.plot(walk_df['time'], walk_loess_smoothed[:, 1], label='LOWESS', color='blue')
	#plt.plot(jump_df['time'], jump_loess_smoothed[:, 1], label='LOWESS', color='green')
	#plt.plot(still_df['time'], still_loess_smoothed[:, 1], label='LOWESS', color='yellow')
	#plt.plot(multi_df['time'], multi_loess_smoothed[:, 1], label='LOWESS', color='orange')
	#plt.show()

### Kalman Filtering
def kalmanSmooth(coef, data, X_columns, y_column, output_file):
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
    #regression_line = regression_model.predict(time)

    #plt.figure(figsize=(15, 6))
    #plt.plot(data['time'], data['speed'], 'b.', alpha=0.5, label='Observed speed')
    #plt.plot(data['time'], kalman_smoothed[:, 0], 'g.', label='Kalman smoothed speed')  # Use scatter for Kalman points
    #plt.plot(data['time'], regression_line, 'r-', label='Linear regression line')
    #plt.xlabel('Time')
    #plt.ylabel('Speed')
    #plt.legend()
    #plt.savefig(outfile)
    #plt.close()

    # Create DataFrame with smoothed data
    smoothed_df = pd.DataFrame({
        'time': data['time'],
        'speed': kalman_smoothed[:, 0],
    })

    # Save the smoothed data
    smoothed_df.to_csv(output_file, index=False)


def output_kalman(df, label):
    output_dir = 'kalman_smoothed'

    X_columns = ['ax', 'ay', 'az', 'speed']
    y_column = 'next_speed'

    # Read file and make new column
    df[y_column] = df['speed'].shift(-1)
    df['time'] = pd.to_datetime(df['time'])
    df = df.dropna()

    # Split data into train and valid sets
    train, valid = train_test_split(df, test_size=0.3, random_state=42)
    X_train, y_train = train[X_columns], train[y_column]
    X_valid, y_valid = valid[X_columns], valid[y_column]

    # Train KNeighborsRegressor model
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)

    print(f"Training score: {model.score(X_train, y_train)}")
    print(f"Validation score: {model.score(X_valid, y_valid)}")

    coefficients = model.predict(X_train[:1]).reshape(-1)
    output_file = os.path.join(output_dir, f'{label}_smoothed.csv')
    kalmanSmooth(coefficients, df, X_columns, y_column, output_file)
