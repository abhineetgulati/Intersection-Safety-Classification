import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


def do_it_all(image_path):
    """
    Full preprocessing pipeline for satellite intersection imagery:
    
    1. Gaussian Blur
    2. Grayscale Conversion
    3. Scharr Edge Detection
    4. Normalize Scharr intensity
    5. Rotation detection using DFT (BEFORE masking)
    6. Rotate EDGE image (not grayscale)
    7. Apply BLACK circular fading mask
    8. Standard scaling to [0,1]

    Returns:
        final (float32): CNN-ready image scaled to [0,1]
    """

    # =====================================================
    # 1. LOAD IMAGE
    # =====================================================
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # =====================================================
    # 2. GAUSSIAN BLUR
    # =====================================================
    blurred = cv2.GaussianBlur(
        img,
        (3, 3),
        0
    )

    # =====================================================
    # 3. GRAYSCALE CONVERSION
    # =====================================================
    gray = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    # =====================================================
    # 4. SCHARR EDGE DETECTION
    # =====================================================
    grad_x = cv2.Scharr(
        gray,
        cv2.CV_64F,
        1,
        0
    )

    grad_y = cv2.Scharr(
        gray,
        cv2.CV_64F,
        0,
        1
    )

    # Use magnitude instead of weighted sum
    magnitude = cv2.magnitude(
        grad_x,
        grad_y
    )

    # =====================================================
    # 5. NORMALIZE SCHARR INTENSITY
    # =====================================================
    scharr_result = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(
        "step_1_scharr_edges.png",
        scharr_result
    )

    # =====================================================
    # ROTATION DETECTION USING HOUGH LINES
    # =====================================================

    # Detect strong lines
    lines = cv2.HoughLinesP(
        scharr_result,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=60,
        maxLineGap=20
    )

    if lines is None:
        print("No lines detected — using 0° rotation")
        coarse_angle = 0

    else:
        angles = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle = np.degrees(
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                )
            )

            # Normalize angle to [-90, 90]
            if angle < -90:
                angle += 180
            elif angle > 90:
                angle -= 180

            # Ignore tiny noisy lines
            length = np.sqrt(
                (x2 - x1) ** 2 +
                (y2 - y1) ** 2
            )

            if length > 80:
                angles.append(angle)

        if len(angles) == 0:
            print("No strong lines found — using 0°")
            coarse_angle = 0

        else:
            # Median is much more robust than mean
            coarse_angle = np.median(angles)

    precise_angle = refine_rotation_angle(
        scharr_result,
        coarse_angle
    )

    print(f"Coarse angle: {coarse_angle:.2f}")
    print(f"Refined angle: {precise_angle:.2f}")

    # =====================================================
    # 10. ROTATE THE SCHARR IMAGE
    # IMPORTANT:
    # Rotate EDGE image, not grayscale
    # =====================================================
    rows, cols = scharr_result.shape

    M = cv2.getRotationMatrix2D(
        (cols / 2, rows / 2),
        precise_angle,
        1.0
    )

    rotated = cv2.warpAffine(
        scharr_result,
        M,
        (cols, rows),

        # Better interpolation for satellite imagery
        flags=cv2.INTER_CUBIC,

        # Fill corners with BLACK
        borderValue=0
    )

    cv2.imwrite(
        "step_3_rotated_edges.png",
        rotated
    )

    # =====================================================
    # 11. APPLY BLACK CIRCULAR FADING MASK
    # =====================================================
    h, w = rotated.shape
    center_y, center_x = h // 2, w // 2

    y, x = np.ogrid[:h, :w]

    dist_from_center = np.sqrt(
        (x - center_x) ** 2 +
        (y - center_y) ** 2
    )

    # Tune these for your dataset
    short_side = min(h, w)

    r_inner = int(short_side * 0.25) # 20% of the image
    r_outer = int(short_side * 0.55) # 50% of the image (touches the edges)

    # Inside inner radius -> keep fully
    # Outside outer radius -> black
    mask = np.clip(
        (dist_from_center - r_inner) /
        (r_outer - r_inner),
        0,
        1
    )

    # Fade to BLACK instead of white
    final = rotated * (1 - mask)

    final = final.astype(np.uint8)

    cv2.imwrite(
        "step_4_final_masked.png",
        final
    )

    # =====================================================
    # 12. STANDARD SCALING FOR CNN
    # =====================================================
    final = final.astype(np.float32) / 255.0

    return final