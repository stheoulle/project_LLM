import os
import json
import re
from docx import Document

# Dossier contenant les fichiers .docx
input_dir = "CDD-CESM/Medical-reports-for-cases-/Medical reports for cases"
output_dir = "CDD-CESM/json_output"

os.makedirs(output_dir, exist_ok=True)

def extract_birads(text):
    match = re.search(r"BIRADS\s*(\d)", text.upper())
    return int(match.group(1)) if match else None

def extract_mass_description(text):
    return re.search(r"(mass.*?margin.*?)\.", text, re.IGNORECASE)

def extract_findings(text, side):
    findings = {}
    side_block = re.search(rf"{side} Breast:(.*?)(?=(Right Breast:|Left Breast:|OPINION|$))", text, re.DOTALL | re.IGNORECASE)
    if side_block:
        block_text = side_block.group(1).strip()
        findings['mass_presence'] = 'mass' in block_text.lower() and 'no mass' not in block_text.lower()
        findings['mass_description'] = extract_mass_description(block_text).group(1) if extract_mass_description(block_text) else None
        findings['skin_retraction'] = 'skin retraction' in block_text.lower() or 'skin indentation' in block_text.lower()
        findings['nipple_retraction'] = 'nipple retraction' in block_text.lower()
        findings['birads'] = extract_birads(block_text)
        findings['raw_text'] = block_text
    return findings

def process_docx(path):
    doc = Document(path)
    full_text = '\n'.join([para.text for para in doc.paragraphs])

    patient_id_match = re.search(r"PATIENT\s*NO\.*\s*(\d+)", full_text, re.IGNORECASE)
    patient_id = patient_id_match.group(1) if patient_id_match else None

    acr_match = re.search(r"ACR\s*[A-D]\s*:\s*.*", full_text)
    acr_density = acr_match.group(0).strip() if acr_match else None

    examinations = []

    # Digital Mammography
    if "DIGITALIZED" in full_text.upper():
        exam = {
            "modality": "Digital Mammography",
            "findings": {
                "right_breast": extract_findings(full_text, "Right"),
                "left_breast": extract_findings(full_text, "Left")
            }
        }
        examinations.append(exam)

    # CESM
    if "CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY" in full_text.upper():
        cesm_block = re.search(r"CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY.*", full_text, re.DOTALL | re.IGNORECASE)
        if cesm_block:
            text = cesm_block.group(0)
            exam = {
                "modality": "Contrast Enhanced Spectral Mammography",
                "findings": {
                    "right_breast": extract_findings(text, "Right"),
                    "left_breast": extract_findings(text, "Left")
                }
            }
            examinations.append(exam)

    return {
        "patient_id": patient_id,
        "acr_density": acr_density,
        "examinations": examinations
    }

# Parcours des fichiers
for filename in os.listdir(input_dir):
    if filename.endswith(".docx"):
        docx_path = os.path.join(input_dir, filename)
        result = process_docx(docx_path)
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"✔️ JSON saved: {json_path}")
