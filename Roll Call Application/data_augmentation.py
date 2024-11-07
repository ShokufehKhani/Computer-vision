import cv2
import numpy as np

def augment_image(image):
    augmented_images = []

    # Original image
    augmented_images.append(image)

    # Flipped images
    augmented_images.append(cv2.flip(image, 1))  # Horizontal flip

    # Rotated images
    for angle in [60, 90, 180, 270]:
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        augmented_images.append(rotated)

    # Brightness and contrast adjustments
    for alpha in [0.8, 1.2]:  # Contrast control (0.8 to 1.2)
        for beta in [-40, 40]:   # Brightness control (-40 to 40)
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            augmented_images.append(adjusted)

    # Translations (shifts)
    for x_shift, y_shift in [(-20, 0), (0, 20)]:
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        translated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        augmented_images.append(translated)

    # Zoom augmentation
    for zoom_factor in [0.8, 1.2]:  # Zoom in (1.2) and zoom out (0.8)
        height, width = image.shape[:2]
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        resized = cv2.resize(image, (new_width, new_height))

        if zoom_factor < 1.0:  # Zoom out
            delta_w = width - new_width
            delta_h = height - new_height
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            color = [0]  # Padding color (black) for grayscale images
            zoomed = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        else:  # Zoom in
            center_x, center_y = new_width // 2, new_height // 2
            half_width, half_height = width // 2, height // 2
            zoomed = resized[center_y - half_height:center_y + half_height, center_x - half_width:center_x + half_width]

        augmented_images.append(zoomed)

    # Perspective transformations
    rows, cols = image.shape[:2]  # Grayscale image
    ch = 1  # Grayscale image

    for points in [
        [[0, 0], [cols, 0], [0, rows], [cols, rows]],          # No change
        [[0, 0], [cols, 0], [0, rows], [cols, rows - 30]],     # Bottom side slanted
        [[30, 0], [cols - 30, 0], [0, rows], [cols, rows]],    # Top side slanted
        [[0, 30], [cols, 0], [0, rows - 30], [cols, rows]],    # Left side slanted
        [[0, 0], [cols, 30], [0, rows], [cols, rows - 30]]     # Right side slanted
    ]:
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(image, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        augmented_images.append(dst)

    return augmented_images