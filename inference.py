# inference.py
import torch
import numpy as np
import rasterio
from rasterio.features import shapes
import cv2
from model import UNet
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

def detect_spill(image_path, model_path="unet_spill_checkpoint.pth", image_size=128):
    device = torch.device("cpu")
    model = UNet(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with rasterio.open(image_path) as src:
        img = src.read(1).astype(np.float32)
        orig_shape = img.shape
        transform = src.transform   # pixel-to-geo-coordinate mapping

    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img_resized = cv2.resize(img_norm, (image_size, image_size))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = (torch.sigmoid(pred) > 0.5).float().squeeze().numpy()

    # Resize back to original image size
    pred_mask_full = cv2.resize(pred_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_mask_full = pred_mask_full.astype(np.uint8)

    # ---- Area calculation ----
    area_pixels = np.count_nonzero(pred_mask_full)
    pixel_size_m = 10   # Sentinel-1 ground resolution ~10m, adjust if different
    area_km2 = area_pixels * (pixel_size_m * pixel_size_m) / 1_000_000

    # ---- Polygon extraction (mask -> GeoJSON) ----
    polygon_geojson = mask_to_geojson(pred_mask_full, transform)

    # ---- Age estimation (rough heuristic) ----
    age_hrs = estimate_age(pred_mask_full)

    return {
        "polygon": polygon_geojson,
        "area_km2": round(area_km2, 2),
        "age_hrs": age_hrs,
        "pixel_count": int(area_pixels)
    }


def mask_to_geojson(mask, transform):
    """Convert a binary mask into a single (possibly multi-part) GeoJSON polygon."""
    if mask.sum() == 0:
        return None   # koi oil detect nahi hua

    polygons = []
    for geom, value in shapes(mask, mask=mask.astype(bool), transform=transform):
        if value == 1:
            polygons.append(shape(geom))

    if not polygons:
        return None

    # Sab chhote polygons ko combine karo ek single shape me
    merged = unary_union(polygons)
    return mapping(merged)   # GeoJSON format


def estimate_age(mask):
    """
    Rough age heuristic based on spread/shape irregularity.
    Fresh spills = compact, round shapes. Older spills = spread out, irregular, thin.
    This is a simplified approximation for demo purposes, not scientifically precise.
    """
    area = np.count_nonzero(mask)
    if area == 0:
        return 0.0

    # Perimeter-to-area ratio: higher ratio = more irregular/spread = older
    from scipy import ndimage
    eroded = ndimage.binary_erosion(mask)
    perimeter = np.count_nonzero(mask) - np.count_nonzero(eroded)

    if area == 0:
        return 0.0

    compactness = perimeter / (area ** 0.5)  # normalized shape irregularity

    # Simple linear mapping (calibrate later with real historical cases if possible)
    estimated_age = min(compactness * 2, 48)   # cap at 48 hours for demo sanity
    return round(estimated_age, 1)


if __name__ == "__main__":
    result = detect_spill("dataset/images/00000.tif")
    print(f"Area: {result['area_km2']} km²")
    print(f"Estimated age: {result['age_hrs']} hrs")
    print(f"Polygon: {result['polygon']}")