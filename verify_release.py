from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'FILES_SHA256.json').read_text())
for name,expected in manifest.items():
    path=root/name
    assert path.is_file(),name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
print(f"Verified {len(manifest)} archived files")
