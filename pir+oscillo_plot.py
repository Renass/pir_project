import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt

'''
Program for visual check 
OSCILLO flower voltage activity together with
PIR (motion sensor) data
'''

OSCILLO_FILE = 'dataset/2023-11-29/2023-11-29_11_01_25.CSV'
PIR_FILE = 'dataset/2023-11-29/2023-11-29 10_57_32pir_log.txt'

if __name__ == '__main__':
    current_directory = os.getcwd()
    oscillo_path = os.path.join(current_directory, OSCILLO_FILE)
    oscillo_v1 = []
    oscillo_v2 = []

    # Specify the encoding
    encoding = 'iso-8859-1'

    with open(oscillo_path, 'r', newline='', encoding=encoding) as file:
        reader = csv.reader(file)

        # Skip the header lines
        for _ in range(23):
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
                oscillo_v2.append((datetime_obj, float(v_ch2)))

    # Separate the datetime and voltage data for oscillo_v1 and oscillo_v2
    datetime_v1, voltage_v1 = zip(*oscillo_v1)
    datetime_v2, voltage_v2 = zip(*oscillo_v2)











    pir_path = os.path.join(current_directory, PIR_FILE)
    with open(pir_path, 'r') as file:
        lines = file.readlines()
    pir_date_time = [datetime.strptime(line.strip(), '%Y-%m-%d %H:%M:%S') for line in lines]
    y_values = [0] * len(pir_date_time)











    # Create a time series plot for oscillo_v1
    plt.figure(figsize=(10, 6))
    plt.plot(datetime_v1, voltage_v1, label='Voltage (CH1)')
    plt.plot(pir_date_time, y_values, 'ro', markersize=2)
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Voltage vs. Time (CH1)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create a time series plot for oscillo_v2
    plt.figure(figsize=(10, 6))
    plt.plot(datetime_v2, voltage_v2, label='Voltage (CH2)')
    plt.plot(pir_date_time, y_values, 'ro', markersize=2)
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Voltage vs. Time (CH2)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
