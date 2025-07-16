import vtk
from vedo import Volume, show, settings

# Optional: Improve rendering quality
settings.use_depth_peeling = True

# --- Load DICOM images ---
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName("Filtered-CMB-BRCA/11-12-1959-MRChest-61832/301.000000-AX T1-21409")
reader.Update()

# --- Convert DICOM to volume ---
volume_data = reader.GetOutput()

# --- Create vedo Volume from VTK image data ---
vol = Volume(volume_data)

# Optional: Customize appearance
vol.color("bone")       # Try: "bone", "jet", "hot", etc.
vol.alpha([0, 0.2, 0.4, 0.6, 0.9])  # Transparency map
vol.cmap("bone")        # Set color map

# --- Show in a 3D scene ---
show(vol, axes=1, bg='black', viewup='z')
