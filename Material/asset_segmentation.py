import os
import numpy as np
import torch
import cv2
import argparse
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils.sam_utils import create, seed_everything, save_gpt_input

import warnings
# Ignore the SAM2 UserWarning if it appears
warnings.filterwarnings("ignore", category=UserWarning)

def sam_image(
        sam2: torch.nn.Module,
        render_images_path: str,
        points_per_side: int = 32,
        points_per_batch: int = 128,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.85,
        stability_score_offset: float = 0.7,
        crop_n_layers: int = 1,
        box_nms_thresh: float = 0.7,
        crop_n_points_downscale_factor: int = 1,
        min_mask_region_area: int = 900,
        use_m2m: bool = False,
    ):
    
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        crop_n_layers=crop_n_layers,
        box_nms_thresh=box_nms_thresh,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        use_m2m=use_m2m,
    )

    for idx, asset in tqdm(enumerate(os.listdir(render_images_path)), desc="Segmenting assets", unit=" asset"):
        asset_path = os.path.join(render_images_path, asset)
        img_folder = os.path.join(asset_path, 'images')
        data_list = sorted(os.listdir(img_folder))
        img_list = []
        alpha_list = []

        for data_path in data_list:
            image_path = os.path.join(img_folder, data_path)
            image_rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            alpha = image_rgba[:, :, 3]

            # Ensure alpha mask is binary
            alpha[alpha < 125] = 0
            alpha[alpha >= 125] = 255

            image = cv2.imread(image_path)
            image = torch.from_numpy(image)

            img_list.append(image)
            alpha_list.append(alpha[None, ...])

        # Prepare images and alphas for processing
        images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
        imgs = torch.cat(images)
        alphas = np.concatenate(alpha_list, 0)

        save_folder = os.path.join(asset_path, 'seg')
        os.makedirs(save_folder, exist_ok=True)

        # Generate segmentation maps
        seg_map_vis = create(imgs, alphas, data_list, save_folder, mask_generator)
    
    sam2.to('cpu')
    del sam2


if __name__ == '__main__':
    device = "cuda"
    sam2_checkpoint = "./sam2/checkpoints/sam2_hiera_base_plus.pt"
    model_cfg = "configs/sam2/sam2_hiera_b+.yaml"
    assets_path = "renders"

    seed_everything(42)

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    sam_image(sam2, assets_path)
    save_gpt_input(assets_path)