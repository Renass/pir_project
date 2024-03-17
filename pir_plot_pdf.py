import matplotlib.pyplot as plt
from datetime import datetime
import os
from matplotlib.backends.backend_pdf import PdfPages

'''
Program for visual check PIR (motion sensor) data 
preparing document in .pdf format
'''

PIR_FOLDER = '2023-09-27'
PIR_FILE = 'r2023-09-27_pir_log.txt'

if __name__ == '__main__':
    current_directory = os.getcwd()
    # Read the data from the pir_log.txt file
    pir_path = os.path.join(current_directory, PIR_FOLDER, PIR_FILE)
    with open(pir_path, 'r') as file:
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

    # Create a PdfPages object to save the subplots to a PDF file
    pdf_pages = PdfPages('pir_data_report.pdf')

    # Plotting action series with multiple subplots on a single page
    num_series = len(action_series_start)

    if num_series > 0:
        fig, axs = plt.subplots(num_series, figsize=(8.5, 30), gridspec_kw={'hspace': 5})

        for i in range(num_series):
            start_idx = action_series_start[i]
            end_idx = action_series_end[i]

            # Extract the datetime and y values for the current action series
            series_dates = date_times[start_idx:end_idx+1]
            y_values = [1] * len(series_dates)

            axs[i].plot(series_dates, y_values, 'ro', markersize=2)
            axs[i].set_xlabel('Date and Time')
            axs[i].set_yticks([])
            axs[i].set_xlim(series_dates[0], series_dates[-1])  # Set x-axis limits for each subplot

            # Customize the x-axis labels to display the day and rotate them for readability
            axs[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M:%S'))
            axs[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels by 45 degrees for each subplot

        # Adjust the layout to prevent overlapping subplots
        plt.tight_layout()

        # Save the figure with subplots to the PDF
        pdf_pages.savefig(fig, bbox_inches='tight')

    else:
        print('No action series with at least 10 triggers found.')

    # Close the PDF file
    pdf_pages.close()

    print("PDF report  saved as 'pir_data_report.pdf'")
