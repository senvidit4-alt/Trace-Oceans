import requests

record_id = "8346860"

response = requests.get(f"https://zenodo.org/api/records/{record_id}")
data = response.json()

files = data["files"]
print(f"Total files available: {len(files)}")
for f in files[:5]:
    print(f["key"], "-", f["size"], "bytes")