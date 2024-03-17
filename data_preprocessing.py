import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np

'''
Take Ossciloscope data in csv and PIR data with .txt
PIR should fully cover Oscillo data by time

Program prepare .pth dataset for binary classification of affected/not affected
flower voltage time series
'''

OSCILLO_FOLDER = 'dataset/2023-11-29'
OSCILLO_FILE = '2023-11-29_11_01_25.CSV'
PIR_FOLDER = OSCILLO_FOLDER
PIR_FILE = '2023-11-29 10_57_32pir_log.txt'
TIME_SERIES_AMOUNT = 200

if __name__ == '__main__':
    current_directory = os.getcwd()
    oscillo_path = os.path.join(current_directory, OSCILLO_FOLDER, OSCILLO_FILE)
    save_path = os.path.join(current_directory, OSCILLO_FOLDER, 'labeled_data.pth')
    oscillo_v1 = []
    oscillo_v2 = []

    # Specify the encoding
    encoding = 'iso-8859-1'

    with open(oscillo_path, 'r', newline='', encoding=encoding) as file:
        reader = csv.reader(file)

        # Skip the header lines
        for _ in range(22):
            next(reader)

        # Iterate through the data and extract voltage and datetime
        for i, row in enumerate(reader):
            if len(row) == 14:
                id, date_str, time_str, stamps, v_ch1, v_ch2, _, _, _, _, _, _, _, _ = row

                # Modify the datetime string to include milliseconds (stamps)
                datetime_str = f"{date_str} {time_str}.{stamps}"

                # Adjust the format string to include milliseconds
                datetime_obj = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S.%f")
                oscillo_v1.append((datetime_obj, float(v_ch1)))
                #oscillo_v2.append((datetime_obj, float(v_ch2)))
            else:
                print('Uknown data format')

    # Separate the datetime and voltage data for oscillo_v1 and oscillo_v2
    datetime_v1, voltage_v1 = zip(*oscillo_v1)
    #datetime_v2, voltage_v2 = zip(*oscillo_v2)
    print('Total voltage notes',len(datetime_v1))










    pir_path = os.path.join(current_directory, PIR_FOLDER, PIR_FILE)
    with open(pir_path, 'r') as file:
        lines = file.readlines()
    datetime_pir = [datetime.strptime(line.strip(), '%Y-%m-%d %H:%M:%S') for line in lines]
    y_values = [0] * len(datetime_pir)






    # Crop the list to fit the desired shape
    total_length = len(voltage_v1) - (len(voltage_v1) % TIME_SERIES_AMOUNT)
    print('tota_length:',total_length)
    datetime_v1 = datetime_v1[:total_length]
    voltage_v1 = voltage_v1[:total_length]

    # Convert the list to a tensor
    datetime_v1_tensor = np.array(datetime_v1).reshape(-1,TIME_SERIES_AMOUNT)
    voltage_v1_tensor = np.array(voltage_v1).reshape(-1,TIME_SERIES_AMOUNT)
    labels = []
    num_labels0 = 0
    num_labels1 = 0


    labeled_voltage_v1 = []
    for i in range(datetime_v1_tensor.shape[0]):
        sample_start = datetime_v1_tensor[i, 0]
        sample_end = datetime_v1_tensor[i, -1]
        label0 = True
        label1 = False
    
        for trig_time in datetime_pir:
            if (sample_start <= trig_time) and (sample_end >= trig_time):
                label0 = False
            #elif (sample_start <= trig_time) and (datetime_v1_tensor[i, datetime_v1_tensor.shape[1]//2]  >= trig_time):
            if (sample_start <= trig_time) and (datetime_v1_tensor[i, 100]  >= trig_time):
                label1 = True                
        if label0 == True:
            labels.append(0)
            num_labels0 += 1
            labeled_voltage_v1.append(voltage_v1_tensor[i])
        elif label1 == True:
            labels.append(1)
            num_labels1 += 1
            labeled_voltage_v1.append(voltage_v1_tensor[i])
    
    print('num_labels:')
    print(num_labels0)
    print(num_labels1)

    labeled_voltage_v1_tensor = torch.tensor(labeled_voltage_v1)
    labels_tensor = torch.tensor(labels)
    data_to_save = {
    'labeled_voltage_v1': labeled_voltage_v1_tensor,
    'labels': labels_tensor
    }

    # Save the data to a file
    torch.save(data_to_save, save_path)
    print('dataset_saved')
    











    
