#Smooth User speed over time
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pykalman import KalmanFilter
lowess = sm.nonparametric.lowess

#Read file
file = pd.read_csv(f"Data/{file_name}")

#Output graph of smoothed speed over time
plt.figure(figsize=(12, 4))
loess_smoothed = lowess(file['speed'], file['time'], frac=0.085)
#plt.plot(file['time'], file['speed'], 'b.', alpha=0.5, label='Raw Data')
plt.plot(file['time'], loess_smoothed[:, 1], 'r-', label='LOWESS')
plt.show()
