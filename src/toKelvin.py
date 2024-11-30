import numpy as np

# Load the original data
data = np.load('Mid1.npy')

# Convert the 4th row (index 3, since Python is zero-indexed) from Celsius to Kelvin
data[3, :] += 273.15

# Save the updated array to a new file
np.save('Mid1_Kelvin.npy', data)

print("Data successfully converted to Kelvin and saved as 'Later1_Kelvin.npy'.")
