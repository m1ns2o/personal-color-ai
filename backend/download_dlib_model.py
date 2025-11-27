import requests
import bz2
import shutil
from pathlib import Path
import sys

def download_file(url, dest_path):
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {dest_path}")

def decompress_bz2(source_path, dest_path):
    print(f"Decompressing {source_path}...")
    with bz2.open(source_path, 'rb') as source, open(dest_path, 'wb') as dest:
        shutil.copyfileobj(source, dest)
    print(f"Decompressed to {dest_path}")

def main():
    models_dir = Path(__file__).parent / "models" / "dlib"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = "shape_predictor_68_face_landmarks.dat"
    model_path = models_dir / model_name
    archive_path = models_dir / f"{model_name}.bz2"
    url = f"http://dlib.net/files/{model_name}.bz2"

    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return

    try:
        download_file(url, archive_path)
        decompress_bz2(archive_path, model_path)
        
        # Cleanup archive
        archive_path.unlink()
        print("Cleanup complete.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
