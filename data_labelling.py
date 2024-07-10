import pandas
import sys

#python data_labelling.py Walk1.csv walk

file_name = sys.argv[1]
label_name = sys.argv[2]

data = pandas.read_csv(f"Data/{file_name}")
selected_columns = ['ax','ay','az','speed']
new_data = data[selected_columns]
new_data['label'] = label_name

#print(new_data)
new_data.to_csv(f"Data/{file_name}", index = False)