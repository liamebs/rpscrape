import zipfile
from datetime import datetime
import os

def create_audit_zip(unmatched_path, missing_path, output_dir="."):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_filename = f"audit_{timestamp}.zip"
    zip_path = os.path.join(output_dir, zip_filename)

    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(unmatched_path, arcname="unmatched.csv")
        zipf.write(missing_path, arcname="missing_races.csv")

    print(f"✅ Created audit zip: {zip_path}")
    return zip_path

# Example usage
create_audit_zip(
    "data/processed/unmatched.csv",
    "data/processed/missing_races.csv",
    output_dir="data/audits"
)
