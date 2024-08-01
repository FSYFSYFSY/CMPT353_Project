import pandas as pd
import sys

#python data_process.py Walk1.csv walk

file_name = sys.argv[1]
label_name = sys.argv[2]

data = pd.read_csv("raw_data/{}".format(file_name))
start_time = pd.to_datetime(data['time'].iloc[0])

data['time'] = pd.to_datetime(data['time'])

# Reset to timestamp and clean the data
data['time'] = ((data['time'] - start_time).dt.total_seconds() // 0.01 * 0.01)
data = data.groupby('time').agg({'ax':'mean', 'ay':'mean','az':'mean','speed':'mean'}).reset_index()

selected_columns = ['time','ax','ay','az','speed']
new_data = data[selected_columns]
new_data['label'] = label_name

#keep rounding
new_data[['time','ax','ay','az']] = new_data[['time','ax','ay','az']].round(3)



#print(new_data)
new_data.to_csv("original_data/{}".format(file_name), index = False)

