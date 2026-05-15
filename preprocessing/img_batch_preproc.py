import os
import cv2
import numpy as np
from tqdm import tqdm

def batch_preprocess(input_folder):
    """
    Processes all images in a folder and saves them
    to a new folder called 'preprocessed_images'
    inside the same parent directory.

    Original filenames are preserved.
    """

    output_folder = os.path.join(
        input_folder,
        "preprocessed_images"
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    valid_extensions = (
        ".png"
    )

    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(valid_extensions)
    ]

    print(f"Found {len(image_files)} images")
    print(f"Saving to: {output_folder}")

    failed_files = []

    for filename in tqdm(image_files):
        try:
            input_path = os.path.join(
                input_folder,
                filename
            )

            output_path = os.path.join(
                output_folder,
                filename
            )

            processed = do_it_all(
                input_path
            )

            save_img = (processed * 255).astype(np.uint8)

            cv2.imwrite(
                output_path,
                save_img
            )

        except Exception as e:
            failed_files.append(
                (filename, str(e))
            )

    print("\nProcessing complete.")

    if failed_files:
        print(f"\nFailed files: {len(failed_files)}")

        for file, error in failed_files[:20]:
            print(file, "->", error)


# =====================================================
# RUN THIS
# =====================================================

if __name__ == "__main__":

    # CHANGE THIS TO YOUR PEN DRIVE PATH
    input_folder = r"D:\nyc_raw_images"

    batch_preprocess(input_folder)
