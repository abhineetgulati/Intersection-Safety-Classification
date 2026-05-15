import cv2
import numpy as np


def refine_rotation_angle(edge_img, coarse_angle):
    """
    Fine-tune rotation angle around Hough estimate.

    Goal:
    maximize alignment of roads to horizontal/vertical axes.
    """

    h, w = edge_img.shape
    center = (w // 2, h // 2)

    best_angle = coarse_angle
    best_score = -1

    # Search ±8 -> +- 15 degrees around Hough result
    for angle in np.arange(
        coarse_angle - 15,
        coarse_angle + 15,
        0.1
    ):

        M = cv2.getRotationMatrix2D(
            center,
            -angle,
            1.0
        )

        rotated = cv2.warpAffine(
            edge_img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderValue=0
        )

        # Recompute gradients after rotation
        gx = cv2.Sobel(
            rotated,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            rotated,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        # Score:
        # strong x/y gradients means lines are axis-aligned
        score = np.sum(np.abs(gx)) + np.sum(np.abs(gy))

        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle
