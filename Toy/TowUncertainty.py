import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create a grid of coordinates
size = 400
x = np.linspace(-2, 2, size)
y = np.linspace(-2, 2, size)
X, Y = np.meshgrid(x, y)

# Calculate distance from center (origin)
distance = np.sqrt(X**2 + Y**2)

# Define the radius of the circular contour
contour_radius = 1.0
contour_width = 0.2  # Width of the blue contour region

# Calculate distance from the contour line
distance_from_contour = np.abs(distance - contour_radius)

# Create white to blue colormap
blues = plt.cm.Blues
colors_list = blues(np.linspace(0, 1, 256))
# Ensure the first color (corresponding to value 0) is pure white
colors_list[0] = [1, 1, 1, 1]
white_blue_cmap = LinearSegmentedColormap.from_list('white_blues', colors_list)

# Create color map based on distance from contour
# Values close to contour (distance_from_contour small) should be blue (high value)
# Values far from contour should be white (low value)
max_distance = 0.15  # Maximum distance for gradient (reduced for tighter gradient)
color_intensity = np.clip(distance_from_contour / max_distance, 0, 1)
# Invert the values so that close to contour = 1 (blue), far from contour = 0 (white)
color_values = 1 - color_intensity

# Create the plot
plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.set_facecolor('white')
plt.imshow(color_values, extent=[-2, 2, -2, 2], origin='lower', cmap=white_blue_cmap, vmin=0, vmax=1)

plt.axis('off')

# Add a circle to show the exact contour position
circle = plt.Circle((0, 0), contour_radius, fill=False, color='darkblue', linewidth=2, linestyle='--')
plt.gca().add_patch(circle)

plt.show()
