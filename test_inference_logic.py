# test_inference_logic.py
import numpy as np
from inference import mask_to_geojson, estimate_age
import rasterio
from rasterio.transform import from_origin

# Ek dummy mask banao jisme beech me ek "spill jaisा" blob ho
dummy_mask = np.zeros((100, 100), dtype=np.uint8)
dummy_mask[40:60, 40:70] = 1   # rectangular blob = fake oil spill

dummy_transform = from_origin(72.5, 19.0, 0.0001, 0.0001)  # dummy geo-coordinates

polygon = mask_to_geojson(dummy_mask, dummy_transform)
age = estimate_age(dummy_mask)

print("Polygon generated:", polygon is not None)
print("Estimated age:", age, "hrs")
print("Polygon:", polygon)
