import matplotlib.pyplot as plt
from datetime import datetime
import os

'''
Program for visual check PIR (motion sensor) data 
'''

PIR_FILE = 'dataset/2023-10-19/2023-10-19 17_15_51pir_log.txt'

if __name__ == '__main__':
    current_directory = os.getcwd()
    # Read the data from the pir_log.txt file
    with open(os.path.join(current_directory, PIR_FILE), 'r') as file:
        lines = file.readlines()

    # Convert date-time strings to datetime objects
    date_times = [datetime.strptime(line.strip(), '%Y-%m-%d %H:%M:%S') for line in lines]

    # Calculate the time intervals between consecutive entries
    time_intervals = [(date_times[i+1] - date_times[i]).total_seconds() for i in range(len(date_times)-1)]

    # Define a threshold (e.g., 30 seconds) for action series separation
    threshold = 60

    # Initialize lists to store action series start and end indices
    action_series_start = []
    action_series_end = []

    # Determine action series start and end indices
    current_start = 0
    for i, interval in enumerate(time_intervals):
        if interval > threshold:
            # Check if the action series has at least 10 triggers
            if i - current_start >= 10:
                action_series_start.append(current_start)
                action_series_end.append(i - 1)
            current_start = i + 1

    # If the last series extends to the end of the data and has at least 10 triggers, add it
    if current_start < len(date_times) - 1 and len(date_times) - current_start >= 10:
        action_series_start.append(current_start)
        action_series_end.append(len(date_times) - 1)

    # Plotting action series with separate subplots and individual x-axis scales
    num_series = len(action_series_start)

    if num_series > 0:
        fig, axs = plt.subplots(num_series, 1, figsize=(12, 2 * num_series), sharex=False, gridspec_kw={'hspace': 1})  # Disable shared x-axis

        for i in range(num_series):
            start_idx = action_series_start[i]
            end_idx = action_series_end[i]

            # Extract the datetime and y values for the current action series
            series_dates = date_times[start_idx:end_idx+1]
            y_values = [1] * len(series_dates)

            axs[i].plot(series_dates, y_values, 'ro', markersize=2)
            #axs[i].set_title(f'Action Series {i+1}')
            axs[i].set_yticks([])
            axs[i].set_xlim(series_dates[0], series_dates[-1])  # Set x-axis limits for each subplot

            # Customize the x-axis labels to display the day and rotate them for readability
            axs[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M:%S'))
            axs[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels by 45 degrees for each subplot

        plt.xlabel('Date and Time')
        plt.tight_layout()  # Adjust subplot spacing for better readability
        plt.show()
    else:
        print('No action series with at least 10 triggers found.')

