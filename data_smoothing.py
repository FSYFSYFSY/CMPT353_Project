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
from scipy.fft import fft, ifft

#Smooth User speed over time for entire list of dataframe
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

def lowess_smooth(df, label, file_name):
    #Get output directory
    output_dir = 'lowess_smoothed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read and smooth processed data
    smoothed_speed = sm.nonparametric.lowess(df['speed'], df['time'], frac=0.085)
    smoothed_ax = sm.nonparametric.lowess(df['ax'], df['time'], frac=0.085)
    smoothed_ay = sm.nonparametric.lowess(df['ay'], df['time'], frac=0.085)
    smoothed_az = sm.nonparametric.lowess(df['az'], df['time'], frac=0.085)

    # Create DataFrame with smoothed data
    smoothed_df = pd.DataFrame({
        'time': df['time'], 
        'ax': np.round(smoothed_ax[:, 1], 3),
        'ay': np.round(smoothed_ay[:, 1], 3), 
        'az': np.round(smoothed_az[:, 1], 3),
        'speed': np.round(smoothed_speed[:, 1], 3), 
        'label': label})

    # Generate the output file name
    output_file = os.path.join(output_dir, f'{file_name}.csv')

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
        'ax': np.round(kalman_smoothed[:, 0], 3),
        'ay': np.round(kalman_smoothed[:, 1], 3),
        'az': np.round(kalman_smoothed[:, 2], 3),
        'speed': np.round(kalman_smoothed[:, 3], 3),
        'label': data['label']
    })

    # Save the smoothed data
    smoothed_df.to_csv(output_file, index=False)


def output_kalman(df, label, filename):
    #Get output directory
    output_dir = 'kalman_smoothed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_columns = ['ax', 'ay', 'az', 'speed']
    y_column = 'next_speed'

    # Read file and make new column
    df[y_column] = df['speed'].shift(-1)
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
    output_file = os.path.join(output_dir, f'{filename}.csv')
    kalmanSmooth(coefficients, df, X_columns, y_column, output_file)

#FFT denoise function
def fft_denoise(signal, threshold):
    signal_fft = fft(signal, axis=0)
    freqs = np.fft.fftfreq(len(signal))
    signal_fft[np.abs(freqs) > threshold] = 0
    cleaned_signal = ifft(signal_fft)
    return cleaned_signal.real

def apply_fft_denoise(df, label, filename):
    #Get output directory
    output_dir = 'fft_denoised'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df['speed'] = np.round(fft_denoise(df['speed'].values, threshold=0.2), 3)
    df['ax'] = np.round(fft_denoise(df['ax'].values, threshold=0.2), 3)
    df['ay'] = np.round(fft_denoise(df['ay'].values, threshold=0.2), 3)
    df['az'] = np.round(fft_denoise(df['az'].values, threshold=0.2), 3)
    df['label'] = label

    # Generate the output file name
    output_file = os.path.join(output_dir, f'{filename}.csv')

    # Save the denoised data to CSV files
    df[['time', 'ax', 'ay', 'az', 'speed','label']].to_csv(output_file, index=False)
