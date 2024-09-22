import numpy as np
import cv2

from tqdm import tqdm


class ShapeGenerator:
    def __init__(self, args, shape_function):
        self.args = args
        self.shape_function = shape_function

    def makeShapes(self):
        for arg in self.args:
            shape = self.shape_function(arg)
            yield arg, np.array(shape)


class ShapeDetector:
    def __init__(self, image: np.ndarray, shapes: ShapeGenerator):
        self.image = image / image.max()
        self.shapes = shapes

        self.max_response = None
        self.max_idx = None

    def match(self):
        param, shape = next(self.shapes.makeShapes())
        self.max_response = cv2.filter2D(self.image, -1, shape)
        self.max_idx = np.zeros((*self.image.shape, len(self.shapes.args[0])))
        for param, shape in tqdm(
            self.shapes.makeShapes(),
            desc="Matching shapes",
            total=len(self.shapes.args),
        ):
            filtered_image = cv2.filter2D(self.image, -1, shape)
            mask = filtered_image > self.max_response
            self.max_response[mask] = filtered_image[mask]
            self.max_idx[mask] = np.array(param)

    def __call__(self, threshold: float) -> np.ndarray:
        """
        threshold: float in [-1, 1]
        """
        if self.max_response is None or self.max_idx is None:
            raise ValueError("Must call match() first.")

        self.mask = np.zeros_like(self.image, dtype=np.int8)
        sorted_intensities = np.sort(self.max_response, axis=None)[::-1]

        num_iterations = sum(corr >= threshold for corr in sorted_intensities)
        for corr in tqdm(
            sorted_intensities, desc="Creating mask", total=num_iterations
        ):

            if corr <= threshold:
                break

            tmp = np.argwhere(self.max_response == corr)  # (row, col)
            params = self.max_idx[tmp[0, 0], tmp[0, 1], :]

            self.shapes.args = [params]
            _, curr_shape = next(self.shapes.makeShapes())

            # Embed shape
            cr, cc = tmp[0, 0], tmp[0, 1]

            half_size = np.array(curr_shape).shape[0] // 2

            if half_size == 0:
                continue

            shape = np.pad(np.zeros_like(self.image), (half_size, half_size))
            shape[cr: cr + 2 * half_size, cc: cc + 2 * half_size] = curr_shape
            shape = shape[half_size:-half_size, half_size:-half_size]

            if ((self.mask > 0) & (shape > 0)).any():
                continue

            self.mask = self.mask + (shape > 0) * 1

        return self.mask
