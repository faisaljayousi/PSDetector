"""

"""
import numpy as np


def getEllipse(e, theta, size: int = 200, fill_value: int = 1):
    r"""
    Example
    --------
    >>> ellipse = getEllipse(0.7, np.pi/4, 10)
    >>> plt.imshow(filled_ellipse)
    """
    size = int(size)

    # Define major / minor axes of ellipse
    a = 1.0
    b = np.sqrt(1 - e**2)

    # Create meshgrid
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))

    # Create ellipse
    s, c = np.sin(theta), np.cos(theta)
    ellipse = ((x * c - y * s) / a)**2 + ((x * s + y * c) / b)**2

    # Fill ellipse
    mask = np.zeros_like(ellipse)
    mask[ellipse <= 1] = 1

    # Inside
    filled_ellipse = np.zeros_like(mask)
    filled_ellipse[mask == 1] = fill_value

    # Outside
    filled_ellipse[mask == 0] = 0

    # Edge
    edge = np.zeros_like(ellipse)
    edge[np.abs(ellipse - 1) < 0.2] = -1

    # Combine inside, outside, and edge
    filled_ellipse[edge == -1] = -fill_value

    return filled_ellipse


def normaliseEllipse(ellipse: np.ndarray) -> np.ndarray:
    r"""
    Parameters:
    -----------
    ellipse : np.ndarray
    """
    positive_indices = ellipse > 0
    negative_indices = ellipse < 0

    ellipse[positive_indices] /= np.count_nonzero(positive_indices)
    ellipse[negative_indices] /= np.count_nonzero(negative_indices)

    return ellipse


def generateEllipse(params):
    ellipse = getEllipse(*params)
    ellipse = normaliseEllipse(ellipse)
    ellipse = np.flipud(np.fliplr(ellipse))
    return ellipse
