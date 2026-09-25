import pandas as pd
import random
import os
from faker import Faker
from fpdf import FPDF

fake = Faker()

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_csv():
    data = []
    for _ in range(50):
        patient_id = f"PID-{random.randint(1000, 9999)}"
        name = fake.name()
        age = random.randint(18, 70)
        gender = random.choice(["Male", "Female"])
        platelets = random.randint(20000, 150000)
        ns1 = random.choice(["Positive", "Negative"])
        igm = random.choice(["Positive", "Negative"])
        igg = random.choice(["Positive", "Negative"])
        
        if ns1 == "Positive" or igm == "Positive":
            diagnosis = "Acute Dengue Infection"
            risk = "High" if platelets < 50000 else "Moderate"
        else:
            diagnosis = "No Active Dengue Infection"
            risk = "Low"
            
        recommendations = "Hydration and rest." if risk != "High" else "Immediate hospitalization, IV fluids, monitor platelet count."
        doctor_notes = fake.sentence(nb_words=15)
        
        data.append([patient_id, name, age, gender, platelets, ns1, igm, igg, diagnosis, risk, recommendations, doctor_notes])
        
    df = pd.DataFrame(data, columns=["Patient ID", "Patient Name", "Age", "Gender", "Platelet Count", "NS1", "IgM", "IgG", "Diagnosis", "Risk Level", "Recommendations", "Doctor Notes"])
    df.to_csv(f"{OUTPUT_DIR}/dengue_reports.csv", index=False)
    return df

def generate_pdfs(df):
    for index, row in df.iterrows():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        pdf.cell(200, 10, txt=f"Medical Report - {row['Patient ID']}", ln=True, align='C')
        pdf.cell(200, 10, txt="="*50, ln=True, align='C')
        
        for col in df.columns:
            pdf.cell(200, 10, txt=f"{col}: {row[col]}", ln=True)
            
        pdf.output(f"{OUTPUT_DIR}/{row['Patient ID']}_report.pdf")

if __name__ == "__main__":
    print("Generating Synthetic Dataset...")
    df = generate_csv()
    generate_pdfs(df)
    print(f"Generated 50 reports in {OUTPUT_DIR}/")
