
import cv2
import numpy as np


def _to_uint8(image):
    """Convert an image array safely to the 0–255 range."""

    image = np.asarray(image)

    if image.size == 0:
        raise ValueError("The image is empty or invalid.")

    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    finite = np.isfinite(image)

    if not np.any(finite):
        raise ValueError("The image contains no valid pixel values.")

    minimum = float(np.min(image[finite]))
    maximum = float(np.max(image[finite]))

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    if minimum >= 0 and maximum <= 1:
        image = image * 255.0
    elif minimum >= 0 and maximum <= 255:
        pass
    elif maximum > minimum:
        image = (image - minimum) / (maximum - minimum) * 255.0
    else:
        image = np.zeros_like(image)

    return np.clip(image, 0, 255).astype(np.uint8)


def extract_features(image, color_order="BGR"):
    """
    Extract measurable visual image-quality features.

    color_order:
        BGR: OpenCV-loaded images
        RGB: PIL/Streamlit-uploaded images
    """

    if image is None:
        raise ValueError("No image was provided.")

    image = _to_uint8(image)

    if image.ndim == 2:
        gray = image

    elif image.ndim == 3 and image.shape[2] == 3:

        if color_order.upper() == "RGB":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif image.ndim == 3 and image.shape[2] == 4:

        if color_order.upper() == "RGB":
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    else:
        raise ValueError("Unsupported image format.")

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    sharpness = float(laplacian.var())

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    high_frequency_variation = float(np.std(laplacian))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "edge_density": edge_density,
        "noise_indicator": high_frequency_variation,
    }


if __name__ == "__main__":
    image = cv2.imread("test.jpg")

    if image is None:
        print("Image not found! Add test.jpg to the project folder.")

    else:
        print("\n=== SpectraAI Image Analysis ===")

        for feature, value in extract_features(image).items():
            print(f"{feature}: {value:.4f}")
