from PIL import Image
import numpy as np

def apply_mask(image_path: str, mask_path: str, output_path: str) -> None:
    image = np.array(Image.open(image_path).convert("RGBA"))
    mask = np.array(Image.open(mask_path).convert("L"))

    mask_norm = mask.astype(np.float32) / 255.0
    masked = image.copy()
    masked[..., :3] = (image[..., :3].astype(np.float32) * mask_norm[..., None]).astype(np.uint8)
    masked[..., 3] = (image[..., 3].astype(np.float32) * mask_norm).astype(np.uint8)

    Image.fromarray(masked, mode="RGBA").save(output_path)
    
if __name__ == "__main__":
    apply_mask("data/outputs/dataset/000590-003011-000443/full_rgb.png", "data/outputs/dataset/000590-003011-000443/vis_mask.png", "output_image.png")