import streamlit as st
from pathlib import Path
import os
import glob
import numpy as np

st.set_page_config(page_title="Breast 3D Viewer", layout="wide")

ROOT = Path(__file__).parent
BREAST_ROOT = ROOT / "Breast-diagnosis/manifest-BbshIhaG7188578559074019493/BREAST-DIAGNOSIS"

st.title("Breast 3D Viewer")
st.write("Select a study and click OK to generate a 3D rendered view of the breasts.")

# helper functions

def find_studies():
    studies = []
    if not BREAST_ROOT.exists():
        print("BREAST_ROOT does not exist")
        return studies
    for manifest in BREAST_ROOT.glob("BreastDx-01-*"):
        bdir = manifest
        print(f"Checking {bdir}")
        if bdir.exists() and bdir.is_dir():
            # check if contains files
            files = list(bdir.rglob("*"))
            if files:
                studies.append((manifest.name, str(bdir)))
                print(f"Found study: {manifest.name} at {bdir}")
    print(f"Found {len(studies)} studies.")
    return studies


@st.cache_data
def load_dicom_series(folder):
    try:
        import pydicom
    except Exception as e:
        raise RuntimeError("pydicom is required to read DICOM files. Install via pip: pip install pydicom")

    files = [p for p in Path(folder).glob("**/*") if p.is_file()]
    dicom_files = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            # heuristics: DICOM files usually contain 'SOPInstanceUID' or PixelData
            dicom_files.append(str(f))
        except Exception:
            continue
    if not dicom_files:
        raise RuntimeError("No DICOM files found in selected folder.")

    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, 'PixelData'):
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise RuntimeError("No readable DICOM image slices in the selected folder.")

    # sort slices by InstanceNumber or ImagePositionPatient (z)
    def slice_z(s):
        if hasattr(s, 'ImagePositionPatient'):
            return float(s.ImagePositionPatient[2])
        if hasattr(s, 'SliceLocation'):
            return float(s.SliceLocation)
        if hasattr(s, 'InstanceNumber'):
            return float(s.InstanceNumber)
        return 0.0

    slices = sorted(slices, key=slice_z)

    # determine target shape (use max found among slices)
    orig_shapes = [tuple(s.pixel_array.shape) for s in slices]
    max_rows = max(r for r, c in orig_shapes)
    max_cols = max(c for r, c in orig_shapes)
    target_shape = (max_rows, max_cols)

    # prepare volume resized to target_shape
    volume = np.zeros((len(slices), target_shape[0], target_shape[1]), dtype=np.float32)

    # get original spacing from first slice (before resize)
    first = slices[0]
    try:
        px = float(getattr(first, 'PixelSpacing', [1.0, 1.0])[0])
        py = float(getattr(first, 'PixelSpacing', [1.0, 1.0])[1])
    except Exception:
        px = py = 1.0

    try:
        if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient') and hasattr(slices[1], 'ImagePositionPatient'):
            z0 = float(slices[0].ImagePositionPatient[2])
            z1 = float(slices[1].ImagePositionPatient[2])
            pz = abs(z1 - z0)
        else:
            pz = float(getattr(first, 'SliceThickness', 1.0))
    except Exception:
        pz = 1.0

    # need skimage for resizing
    try:
        from skimage.transform import resize
    except Exception:
        raise RuntimeError("scikit-image is required to resize DICOM slices. Install via pip: pip install scikit-image")

    for i, s in enumerate(slices):
        arr = s.pixel_array.astype(np.float32)
        # apply rescale if present
        slope = float(getattr(s, 'RescaleSlope', 1.0))
        intercept = float(getattr(s, 'RescaleIntercept', 0.0))
        arr = arr * slope + intercept

        if arr.shape != target_shape:
            # resize while preserving intensity range
            try:
                arr_resized = resize(arr, target_shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
            except Exception:
                # fallback: center-pad or crop if resize fails
                th, tw = target_shape
                h, w = arr.shape
                out = np.zeros((th, tw), dtype=np.float32)
                y0 = max((th - h) // 2, 0)
                x0 = max((tw - w) // 2, 0)
                y1 = y0 + min(h, th)
                x1 = x0 + min(w, tw)
                oy0 = max((h - th) // 2, 0)
                ox0 = max((w - tw) // 2, 0)
                out[y0:y1, x0:x1] = arr[oy0:oy0+(y1-y0), ox0:ox0+(x1-x0)]
                arr_resized = out
            arr = arr_resized

        volume[i, :, :] = arr

    # adjust spacing to reflect resizing so physical dimensions remain correct
    try:
        new_px = px * (first.Rows / float(target_shape[0]))
        new_py = py * (first.Columns / float(target_shape[1]))
    except Exception:
        new_px, new_py = px, py

    spacing = (pz, new_px, new_py)  # match (z, x, y) order of volume

    return volume, spacing


def generate_mesh(volume, spacing, level=None):
    try:
        from skimage import measure
    except Exception:
        raise RuntimeError("scikit-image is required for mesh extraction. Install via pip: pip install scikit-image")

    if level is None:
        # choose a threshold: take midpoint between min and max
        level = float((np.nanmin(volume) + np.nanmax(volume)) / 2.0)

    # marching cubes expects volume in z, y, x; our volume is (z, rows, cols) so it's fine
    verts, faces, normals, values = measure.marching_cubes(volume, level=level, spacing=spacing)
    return verts, faces


def plot_mesh_plotly(verts, faces):
    try:
        import plotly.graph_objects as go
    except Exception:
        raise RuntimeError("plotly is required for 3D visualization. Install via pip: pip install plotly")

    x, y, z = verts.T
    i, j, k = faces.T
    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.6, color='lightpink')
    fig = go.Figure(data=[mesh])
    fig.update_layout(scene=dict(aspectmode='auto'), margin=dict(l=0, r=0, t=0, b=0))
    return fig


# UI
studies = find_studies()
if not studies:
    st.warning("No studies found under: {}\nMake sure folders are present and contain DICOM files.".format(BREAST_ROOT))
else:
    study_names = [s[0] for s in studies]
    sel = st.selectbox("Select study", options=["-- choose --"] + study_names)

    # Global "Preprocess ALL" controls
    st.markdown("**Preprocess ALL studies**")
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        all_target_spacing = st.number_input("Target spacing for all (mm/px)", value=0.05, format="%.4f", key='all_pp_spacing')
    with col_b:
        all_patch_size = st.number_input("Patch size for all (px, 0 to skip)", value=512, step=32, key='all_pp_patch_size')
    with col_c:
        all_patch_stride = st.number_input("Patch stride for all (px)", value=256, step=32, key='all_pp_patch_stride')
    all_remove_pect = st.checkbox("Remove pectoral (all)", value=False, key='all_pp_remove_pect')

    if st.button("Preprocess ALL studies", key='pp_all_btn'):
        out_root = ROOT / 'preprocessed_all'
        out_root.mkdir(parents=True, exist_ok=True)
        try:
            import sys
            sys.path.insert(0, str(ROOT))
            from preprocess import process_directory
        except Exception as e:
            st.error(f"Failed to import preprocessing utilities: {e}")
        else:
            ps = int(all_patch_size) if int(all_patch_size) > 0 else None
            stride = int(all_patch_stride) if int(all_patch_stride) > 0 else None
            total = len(studies)
            progress = st.progress(0)
            status_box = st.empty()
            for idx, (name, p) in enumerate(studies):
                status_box.text(f'[{idx+1}/{total}] Preprocessing {name}...')
                out_dir = out_root / name
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    process_directory(p, str(out_dir), target_spacing=float(all_target_spacing), patch_size=ps, patch_stride=stride, remove_pectoral=bool(all_remove_pect))
                    status_box.text(f'[{idx+1}/{total}] Done: {name}')
                except Exception as e:
                    status_box.text(f'[{idx+1}/{total}] Failed: {name} — {e}')
                progress.progress((idx + 1) / total)
            status_box.text('All preprocessing tasks finished.')

    # initialize session state keys
    if 'volume' not in st.session_state:
        st.session_state['volume'] = None
    if 'spacing' not in st.session_state:
        st.session_state['spacing'] = None
    if 'selected_study' not in st.session_state:
        st.session_state['selected_study'] = None

    # Load when OK is pressed and persist into session_state
    if st.button("OK", key='ok_btn'):
        if sel == "-- choose --":
            st.warning("Please select a study first.")
        else:
            # find path for selection
            path = None
            for name, p in studies:
                if name == sel:
                    path = p
                    break
            if path is None:
                st.error("Study path not found.")
            else:
                with st.spinner("Loading DICOM series..."):
                    try:
                        volume, spacing = load_dicom_series(path)
                        st.session_state['volume'] = volume
                        st.session_state['spacing'] = spacing
                        st.session_state['selected_study'] = sel
                        st.session_state['selected_path'] = path
                        st.success(f"Loaded study: {sel} (volume shape: {volume.shape})")
                    except Exception as e:
                        st.error(f"Failed to load DICOM series: {e}")
                        st.stop()

    # If we have a loaded volume in session state, use it
    if st.session_state.get('volume') is not None:
        volume = st.session_state['volume']
        spacing = st.session_state['spacing']
        selected_path = st.session_state.get('selected_path')

        st.write("Selected study:", st.session_state.get('selected_study'))
        st.write("Volume shape (slices, rows, cols):", volume.shape)

        # Per-study Preprocessing controls
        st.markdown("**Preprocessing (run on selected study)**")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            target_spacing = st.number_input("Target spacing (mm/px)", value=0.05, format="%.4f", key='pp_spacing')
        with col2:
            patch_size = st.number_input("Patch size (px, 0 to skip)", value=512, step=32, key='pp_patch_size')
        with col3:
            patch_stride = st.number_input("Patch stride (px)", value=256, step=32, key='pp_patch_stride')
        remove_pect = st.checkbox("Remove pectoral muscle", value=False, key='pp_remove_pect')

        if st.button("Preprocess study", key='pp_btn'):
            if not selected_path:
                st.error('Selected path not available')
            else:
                out_dir = ROOT / 'preprocessed' / st.session_state.get('selected_study', 'study')
                out_dir.mkdir(parents=True, exist_ok=True)
                with st.spinner('Running preprocessing (may take a while)...'):
                    try:
                        import sys
                        sys.path.insert(0, str(ROOT))
                        from preprocess import process_directory
                        ps = int(patch_size) if int(patch_size) > 0 else None
                        stride = int(patch_stride) if int(patch_stride) > 0 else None
                        process_directory(selected_path, str(out_dir), target_spacing=float(target_spacing), patch_size=ps, patch_stride=stride, remove_pectoral=bool(remove_pect))
                        st.success(f'Preprocessing finished. Outputs in: {out_dir}')
                    except Exception as e:
                        st.error(f'Preprocessing failed: {e}')

        vmin = float(np.nanmin(volume))
        vmax = float(np.nanmax(volume))
        # use a persistent slider value in session_state via key
        threshold = st.slider("Surface level (threshold)", vmin, vmax, (vmin + vmax) / 2.0, key='threshold')

        if st.button("Render 3D", key='render_btn'):
            with st.spinner("Extracting mesh (marching cubes)... this may take a while"):
                try:
                    verts, faces = generate_mesh(volume, spacing, level=threshold)
                    fig = plot_mesh_plotly(verts, faces)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Rendering failed: {e}")

        # show middle slice preview
        mid = volume.shape[0] // 2
        st.subheader("Middle slice preview")
        import matplotlib.pyplot as plt
        fig2, ax = plt.subplots()
        ax.imshow(volume[mid, :, :], cmap='gray')
        ax.axis('off')
        st.pyplot(fig2)
    else:
        st.info("No study loaded. Select a study and click OK to load the DICOM series.")


# footer
st.write("\nNotes: This viewer uses pydicom, scikit-image and plotly. If missing, install them in your environment. For large volumes the mesh extraction can be memory/CPU-intensive.")
