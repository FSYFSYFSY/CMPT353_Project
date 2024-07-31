**Required Libaries:**

```
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pykalman import KalmanFilter
from scipy.fft import fft, ifft

#This is the data smoothing file used
from data_smoothing import output_kalman, kalmanSmooth, lowess_smooth, fft_denoise, apply_fft_denoise
```

**Command used**

The data in the Data folder is already processed but not yet cleaned, the command used for processing is python3 data_process.py file_name label_name(The motion category for the data). The rest of code is in the program.ipynb, and no more commands are used.

**Order of Execution**

Starting fresh, the order of execution would be to first process the data with data_process.py and the revelant command. Then the rest is in the program.ipynb, starting from loading the process data. Plotting the data and then smoothing the data with different algorithms, and then using different machine learning models to predict the user motion.

**File Produced**

FOr the data smoothing part, the code will read in the directory of Data folder, and output smoothed data files by different smoothing algorithms to different folders. The folders will need to be created before hand. 

