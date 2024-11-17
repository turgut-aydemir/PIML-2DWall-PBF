import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_csv_files(directory):
    # Define the output directory
    output_dir = os.path.join(directory, 'output')
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Initialize lists to collect data for each row
    x_list = []
    y_list = []
    time_list = []
    temp_list = []

    # Time increment for each frame (30 FPS)
    time_increment = 1 / 30.0

    # Initialize the time counter
    current_time = 0.0

    # Define the x values (fixed to -27 to 27)
    x_values = np.arange(-27, 28)  # X values from -27 to 27

    # Loop through each CSV file in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)

            try:
                # Read the CSV file
                df = pd.read_csv(file_path, header=None)

                # Process each row from 172 to 226
                for row_idx in range(172, 227):
                    # Extract values from the row considering columns 124 to 178
                    row_data = df.iloc[row_idx - 1, 123:178].values
                    x = x_values[row_idx - 172]  # X value from -27 to 27

                    # Collect data for the current row
                    for col_idx, temperature in enumerate(row_data):
                        y = col_idx  # Y value starting from 0 to 54
                        x_list.append(x)
                        y_list.append(y)
                        time_list.append(format(current_time, '.7f'))
                        temp_list.append(temperature)

                # Update the time counter for the next file
                current_time += time_increment

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Convert lists to NumPy arrays
    x_array = np.array(x_list, dtype=np.float64)
    y_array = np.array(y_list, dtype=np.float64)
    time_array = np.array(time_list, dtype=np.float64)
    temp_array = np.array(temp_list, dtype=np.float64)

    # Combine arrays into a single array with shape (4, N)
    final_output = np.vstack((x_array, y_array, time_array, temp_array))

    # Get the current date and time
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")

    # Define the output file path with date and time
    output_file = os.path.join(output_dir, f"Processed_Temperature_Data_{date_time_str}.npy")

    # Save the final output to an .npy file
    np.save(output_file, final_output)
    print(f"All files processed. Output saved to {output_file}")

# Usage
directory = r"C:\Users\aydemirt\Desktop\Frame 143306 to 143592"
process_csv_files(directory)
