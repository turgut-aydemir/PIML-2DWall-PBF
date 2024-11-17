import numpy as np
import pandas as pd

# Load the .npy file
npy_file_path = r'C:\Users\aydemirt\Desktop\Frame 143306 to 143592\output\Processed_Temperature_Data_20240910_131639.npy'
data = np.load(npy_file_path)

# Convert the data to DataFrame
df = pd.DataFrame(data.T, columns=['X Values', 'Y Values', 'Time Values', 'Temperature Values'])

# Filter the DataFrame to get all rows where Time Values is 0.0000000
df_t0 = df[df['Time Values'] == 2,8666667]

# Convert the filtered DataFrame back to numpy array
filtered_data = df_t0.values.T

# Save the filtered data to a new .npy file
output_npy_path = r'C:\Users\aydemirt\Desktop\Frame 143306 to 143592\output\Processed_Temperature_Data_20240910_131639_t0.npy'
np.save(output_npy_path, filtered_data)

print(f"Filtered data where Time Values = 0.0000000 has been saved to {output_npy_path}")
