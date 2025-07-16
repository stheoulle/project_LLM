import os
import csv
import pydicom

BASE_DIR = "Filtered-CMB-BRCA"
OUTPUT_CSV = "dataset.csv"

def find_series_dirs(base_dir):
    series_dirs = []
    for root, dirs, files in os.walk(base_dir):
        # On cherche les dossiers qui contiennent des fichiers .dcm
        dcm_files = [f for f in files if f.endswith('.dcm')]
        if dcm_files:
            series_dirs.append(root)
    return series_dirs

def extract_metadata(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        patient_id = ds.get("PatientID", "Unknown")
        study_desc = ds.get("StudyDescription", "Unknown")
        series_desc = ds.get("SeriesDescription", "Unknown")
        return patient_id, study_desc, series_desc
    except Exception as e:
        print(f"Erreur lecture DICOM {dicom_path}: {e}")
        return "Unknown", "Unknown", "Unknown"

def generate_csv(base_dir, output_csv):
    series_dirs = find_series_dirs(base_dir)
    print(f"{len(series_dirs)} séries trouvées.")
    
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["PatientID", "StudyDescription", "SeriesDescription", "ExampleDICOMPath"])
        
        for series_dir in series_dirs:
            # Prendre un exemple de fichier DICOM dans le dossier
            dicom_files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
            if not dicom_files:
                continue
            
            example_dicom = os.path.join(series_dir, dicom_files[0])
            patient_id, study_desc, series_desc = extract_metadata(example_dicom)
            
            writer.writerow([patient_id, study_desc, series_desc, example_dicom])

    print(f"Fichier {output_csv} généré avec succès.")

if __name__ == "__main__":
    generate_csv(BASE_DIR, OUTPUT_CSV)
