import numpy as np
import matplotlib.pyplot as plt

# Constants
wavelength = 632.8e-9  # wavelength of laser light in meters (example value)
vacuum_index = 1.0     # refractive index of vacuum (assuming vacuum)

# Provided fringe shift measurements (replace with your actual data)
fringe_shifts = [68, 66, 63, 70, 56, 56, 54, 69, 72, 66, 59, 55, 43]  # Delta x in nanometers

# Convert fringe shifts to meters
fringe_shifts_m = np.array(fringe_shifts) * 1e-9  # convert to meters

# Calculate electron density (ne) at each point
ne = (2 * fringe_shifts_m * vacuum_index) / (wavelength)

# Create x coordinates for plotting (just using indices)
x_coords = np.arange(len(fringe_shifts))

# Create a colormap based on electron density values
colors = plt.cm.viridis(ne / np.max(ne))  # normalize to [0, 1]

# Plotting electron density (ne) as a heatmap
plt.figure(figsize=(10, 6))
plt.bar(x_coords, ne, color=colors)
plt.xlabel('Index')
plt.ylabel('Electron Density (ne)')
plt.title('1D Distribution of Plasma Density (Heatmap)')
plt.tight_layout()
plt.show()
