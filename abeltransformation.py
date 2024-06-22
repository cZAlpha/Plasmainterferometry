import numpy as np
import abel
import matplotlib.pyplot as plt

# Example list of differences (radial positions)
differences = [68, 66, 63, 70, 56, 56, 54, 69, 72, 66, 59, 55, 43]

# Assume a radial spacing (e.g., 1 unit per difference)
radial_positions = np.arange(len(differences))  # Assuming each difference is 1 unit of radial distance

# Example intensity profile (Gaussian distribution as an example)
intensity_profile = np.exp(-(radial_positions / len(radial_positions))**2)

# Perform inverse Abel transformation using Hansen-Law method
inv_abel = abel.hansenlaw.hansenlaw_transform(intensity_profile, direction='inverse', verbose=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(radial_positions, inv_abel, label='Radial emissivity (after inverse Abel)')
plt.xlabel('Radial position')
plt.ylabel('Intensity / Emissivity')
plt.legend()
plt.title('Inverse Abel Transformation for Plasma Density Estimation')
plt.grid()
plt.show()
