import matplotlib.pyplot as plt

from ellipses import *
from main import *
from scipy.ndimage import median_filter


# Example parameters
eccentricities = np.arange(0.1, 1, 0.05)
thetas = np.arange(0, np.pi, np.pi/8)
sizes = [12, 16, 22, 26, 30, 34, 36]

ellipse_params = [(ecc, theta, size)
                  for ecc in eccentricities for theta in thetas for size in sizes]

# Create an instance of the ShapeGenerator class with the ellipse generator function
generator = ShapeGenerator(
    ellipse_params, shape_function=generateEllipse)


# Example
image = plt.imread('fibronectin.png')
image = median_filter(image, size=11)

sd = ShapeDetector(image, generator)
sd.match()
mask = sd(0.2)

plt.imshow(mask, cmap='gray')
plt.show()
