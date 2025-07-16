import os
import pydicom
import numpy as np

def find_dicom_series(root_path):
    """
    Recursively find all DICOM series folders under the root_path.
    A folder is considered a series if it contains at least one .dcm file.
    """
    series_paths = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dcm_files = [f for f in filenames if f.lower().endswith('.dcm')]
        if len(dcm_files) > 0:
            series_paths.append(dirpath)
    return series_paths


def load_dicom_series(folder_path):
    """
    Load a DICOM series from a folder into a 3D NumPy array.
    """
    slices = []
    for file in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, file)
        if file.endswith(".dcm"):
            ds = pydicom.dcmread(path)
            slices.append(ds.pixel_array)

    volume = np.stack(slices, axis=0)  # shape: (depth, height, width)
    volume = volume.astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume

base_path = "Filtered-CMB-BRCA"
series_paths = find_dicom_series(base_path)

print(f"Found {len(series_paths)} series.")
for path in series_paths[:1]:  # Load the first one for now
    print(f"Loading series at: {path}")
    volume = load_dicom_series(path)
    print("Volume shape:", volume.shape)

    # Optional: visualize one slice
    import matplotlib.pyplot as plt
    plt.imshow(volume[volume.shape[0] // 2], cmap='gray')
    plt.title("Middle Slice")
    plt.axis('off')
    plt.show()
