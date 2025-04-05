from datetime import datetime
import os
import shutil

def run_tracktempo_starter_in_notebook(base_dir="production"):
    today = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join(base_dir, today)
    os.makedirs(folder_path, exist_ok=True)

    assets = {
        "tracktempo_logo_snazzy.png": "tracktempo_logo.png",
        "tracktempo_branding.md": "tracktempo_branding.md",
        "race_modeling_keywords.pdf": "tracktempo_keywords.pdf",
        "feature_extractor_stub.py": "feature_extractor_stub.py"
    }

    copied_files = []
    for src, dst in assets.items():
        src_path = f"/mnt/data/{src}"
        dst_path = os.path.join(folder_path, dst)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            copied_files.append(dst_path)
        else:
            print(f"⚠️ Missing: {src_path}")

    # Create a new empty daily log
    log_path = os.path.join(folder_path, "tracktempo_daily_log.md")
    with open(log_path, "a") as log_file:
        log_file.write(f"# TrackTempo Daily Log\n\n📅 {today}\n\n---\n\n")

    print(f"📁 Initialized TrackTempo daily folder: {folder_path}")
    print("📦 Copied assets:")
    for f in copied_files:
        print(f" - {f}")
    print(f"🗒️  Log created: {log_path}")

    return folder_path
