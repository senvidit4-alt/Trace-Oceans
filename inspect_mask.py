import os
import warnings
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

# Suppress georeference warnings for non-georeferenced raw mask rasters
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

mask_folder = r"C:\Users\Lenovo\Downloads\01_Train_Val_Oil_Spill_mask"

# If extracted archive contains the Mask_oil subfolder, point to it
if os.path.isdir(os.path.join(mask_folder, "Mask_oil")):
    mask_folder = os.path.join(mask_folder, "Mask_oil")

# Filter only image files
valid_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
files = sorted([f for f in os.listdir(mask_folder) if f.lower().endswith(valid_exts)])

print("Mask folder:", mask_folder)
print("Total mask files:", len(files))

if not files:
    print("No mask image files found.")
else:
    print("Sample filenames:", files[:5])

    first_mask = os.path.join(mask_folder, files[0])
    with rasterio.open(first_mask) as src:
        mask = src.read(1)
        print("\n--- Mask Info ({}) ---".format(files[0]))
        print("Shape:", mask.shape)
        print("Data type:", mask.dtype)
        print("Unique values:", np.unique(mask))
        print("Min/Max:", mask.min(), mask.max())
        nonzero = np.count_nonzero(mask)
        total = mask.size
        print(f"Non-zero pixels: {nonzero:,} / {total:,} ({nonzero / total * 100:.2f}%)")