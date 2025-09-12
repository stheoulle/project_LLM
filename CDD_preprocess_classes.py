import os
import shutil
import pandas as pd

def organize_images_by_class(
    images_dir="/home/light/Documents/Perso/Internship/project/CDD-CESM/PKG - CDD-CESM/CDD-CESM",
    annotations_path="/home/light/Documents/Perso/Internship/project/CDD-CESM/Radiology-manual-annotations.xlsx",
    output_base_dir="/home/light/Documents/Perso/Internship/project/CDD-CESM/organized_images"
):
    # Subdirectories containing images
    subdirs = [
        "Low energy images of CDD-CESM",
        "Subtracted images of CDD-CESM"
    ]
    # Read annotations
    df = pd.read_excel(annotations_path)
    class_map = {
        'benign': 'benign',
        'malignant': 'malignant',
        'normal': 'normal'
    }
    # Create output directories
    for cls in class_map.values():
        os.makedirs(os.path.join(output_base_dir, cls), exist_ok=True)
    # Iterate over annotation rows
    for _, row in df.iterrows():
        image_name = str(row['Image_name'])
        label = str(row['Pathology Classification/ Follow up']).lower()
        target_dir = class_map.get(label)
        if not target_dir:
            print(f"Unknown label for {image_name}: {label}")
            continue
        # Search for image in both subdirectories
        found = False
        for subdir in subdirs:
            image_path = os.path.join(images_dir, subdir, image_name + ".jpg")
            if os.path.isfile(image_path):
                shutil.copy(image_path, os.path.join(output_base_dir, target_dir, image_name + ".jpg"))
                print(f"Copied {image_name} to {target_dir}")
                found = True
                break
        if not found:
            print(f"Image not found: {image_name}")

# Example usage:
organize_images_by_class()