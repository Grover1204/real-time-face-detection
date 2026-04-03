import cv2
import os


def load_image(image_path):
    """
    Loads an image from the specified path.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    # Convert BGR to RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image, save_path):
    """
    Saves an RGB image to the specified path.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Convert RGB back to BGR for saving
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr_image)


def crop_image(image, bbox):
    """
    Crops an image given a bounding box.
    bbox: tuple of (xmin, ymin, width, height)
    """
    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height

    h, w, _ = image.shape

    # Ensure coordinates are within image bounds
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    return image[ymin:ymax, xmin:xmax]


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box on the image.
    bbox: tuple of (xmin, ymin, width, height)
    """
    img_copy = image.copy()
    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height

    cv2.rectangle(
        img_copy, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness
    )
    return img_copy
