import argparse
import json
import os

import pandas as pd
import google.generativeai as genai

from dotenv import load_dotenv
from PIL import Image
import pytesseract

load_dotenv()

# If you're on Windows and Tesseract isn't on PATH, uncomment and update this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# --- 1. DataFrame Template Function ---

def create_medical_record_template():
    """
    Creates an empty DataFrame with columns matching the form.
    """
    columns = [
        "Patient Name",
        "Patient D.O.B.",
        "Doctors Prognosis",
        "Dossage",
        "Duration of Treatment",
        "Health Card Number",
        "Visit Date",
        "Medication Recommended",
        "Frequency",
        "Date of Return",
        "Doctors Notes and Further Recommendations",
    ]
    df = pd.DataFrame(columns=columns)
    return df


# --- 2. OCR Function ---

def process_image_to_text(image_path: str) -> str:
    """
    Runs OCR on an image file and returns raw text.
    """
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng")
    return text


# --- 3. Gemini Extraction Function ---

def extract_form_data_with_gemini(client_model, ocr_text, form_fields):
    """
    Uses the Gemini API to extract structured data from raw OCR text.
    """
    prompt = f"""
    You are an AI assistant specialized in medical data entry.
    Extract the following fields from the text provided and return *only* a single,
    valid JSON object. Do not include any other text, explanations, or markdown.
    
    The fields to extract are:
    {', '.join(form_fields)}
    
    If a field is not found, use "N/A" as the value.
    
    Here is the raw text:
    ---
    {ocr_text}
    ---
    """

    try:
        response = client_model.generate_content(prompt)
        json_string = (
            response.text.strip()
            .replace("```json", "")
            .replace("```", "")
        )
        data_dict = json.loads(json_string)
        return data_dict

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from Gemini response.")
        print("Raw response:", response.text)
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# --- 4. Mock Model (fallback if no API key) ---

class MockModel:
    def generate_content(self, prompt):
        class MockResponse:
            text = """
            {
                "Patient Name": "Jane A. Doe",
                "Health Card Number": "1234 567 890",
                "Patient D.O.B.": "1990-05-15",
                "Visit Date": "2025-11-15",
                "Doctors Prognosis": "Common cold",
                "Medication Recommended": "Amoxicillin",
                "Dosage": "500mg",
                "Frequency": "Twice a day",
                "Duration of Treatment": "5 days",
                "Date of Return": "2025-11-20",
                "Doctors Notes and Further Recommendations": "Patient advised to rest and drink plenty of fluids."
            }
            """
        return MockResponse()


# --- 5. Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read a doctor's note image, OCR it, and extract fields with Gemini."
    )
    parser.add_argument(
        "image_path",
        help="Path to the doctor's note image (e.g., note.jpg, note.png)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: image file not found: {args.image_path}")
        exit(1)

    # Configure Gemini
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-flash")
        print("Using real Gemini model.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        print("Falling back to mock model.")
        model = MockModel()

    # Create DataFrame with template columns
    medical_records_df = create_medical_record_template()
    form_fields = list(medical_records_df.columns)

    # --- OCR Step ---
    print("\n--- 1. Running OCR on image ---")
    raw_ocr_text = process_image_to_text(args.image_path)
    print("OCR text:")
    print(raw_ocr_text)

    # --- Gemini Extraction ---
    print("\n--- 2. Sending OCR text to Gemini ---")
    extracted_data = extract_form_data_with_gemini(model, raw_ocr_text, form_fields)

    if extracted_data:
        print("\n--- 3. Gemini Extracted This Data ---")
        print(json.dumps(extracted_data, indent=2))

        # Add record to DataFrame
        new_record_df = pd.DataFrame([extracted_data])
        medical_records_df = pd.concat([medical_records_df, new_record_df], ignore_index=True)

        print("\n--- 4. Final DataFrame ---")
        print(medical_records_df)

        # ---- SAVE AS CSV INSTEAD OF EXCEL ----
        medical_records_df.to_csv("medical_prescription_records.csv", index=False)
        print("\nSaved to medical_prescription_records.csv")

    else:
        print("Could not extract data.")

    print(f"\nPandas version: {pd.__version__}")
