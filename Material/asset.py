import os
import hydra
from tqdm import tqdm
import numpy as np
from asset_visualiser import render_views
from asset_segmentation import sam_image
from omegaconf import DictConfig, OmegaConf
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils.sam_utils import create, seed_everything, save_gpt_input

import logging
logging.getLogger().setLevel(logging.ERROR)

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()

class AssetsMaterialPipeline:
    def __init__(
            self,
            GLB_folder_path: str,
            config: DictConfig
        ): 
        self.path = GLB_folder_path
        self.config = config
        self.render_asset = self.config.RENDER.out_dir

    def asset_visualiser(self) -> None:
        GLB_folder_path = self.path
        asset_path_list = os.listdir(GLB_folder_path)
        assert np.all([asset_path.endswith('.glb') for asset_path in asset_path_list]), "All assets must be in GLB format"
        
        for asset_name in asset_path_list:
            if os.path.splitext(asset_name)[0].endswith("_rotated"):
                asset_path_list.remove(asset_name) 

        azimuth_angles = self.config.RENDER.azimuth_angles
        elevation_angles = self.config.RENDER.elevation_angles
        trillis_asset = self.config.RENDER.trillis_asset
        replace_org_file = self.config.RENDER.replace_org_file
        fov_deg = self.config.RENDER.fov_deg
        resolution = self.config.RENDER.resolution
        mark = self.config.RENDER.mark

        for _, asset_path in tqdm(enumerate(asset_path_list), desc="Rendering Asset Images"):
            render_views(
                glb_path=os.path.join(GLB_folder_path, asset_path),
                out_dir=self.render_asset,
                azimuth_angles=azimuth_angles,
                elevation_angles=elevation_angles,
                trillis_asset=trillis_asset,
                replace_org_file=replace_org_file,
                fov_deg=fov_deg,
                resolution=resolution,
                mark=mark,
            )

    def asset_segmentation(self) -> None:

        device = self.config.SEGMENTATION.device
        sam2_checkpoint = self.config.SEGMENTATION.sam2_checkpoint
        model_cfg = self.config.SEGMENTATION.model_cfg

        points_per_side = self.config.SEGMENTATION.points_per_side
        points_per_batch = self.config.SEGMENTATION.points_per_batch
        pred_iou_thresh = self.config.SEGMENTATION.pred_iou_thresh
        stability_score_thresh = self.config.SEGMENTATION.stability_score_thresh
        stability_score_offset = self.config.SEGMENTATION.stability_score_offset
        crop_n_layers = self.config.SEGMENTATION.crop_n_layers
        box_nms_thresh = self.config.SEGMENTATION.box_nms_thresh
        crop_n_points_downscale_factor = self.config.SEGMENTATION.crop_n_points_downscale_factor
        min_mask_region_area = self.config.SEGMENTATION.min_mask_region_area
        use_m2m = self.config.SEGMENTATION.use_m2m

        # if model_cfg == "sam2_hiera_b+":
        #     model_cfg = "sam2_hiera_b+.yaml"
        # elif model_cfg == "sam2_hiera_l":
        #     model_cfg = "sam2_hiera_l.yaml"
        # elif model_cfg == "sam2_hiera_s":
        #     model_cfg = "sam2_hiera_s.yaml"
        # elif model_cfg == "sam2_hiera_t":
        #     model_cfg = "sam2_hiera_t.yaml"
        # model_cfg = "configs/sam2/sam2_hiera_b+.yaml"
        sam2 = build_sam2(
            model_cfg,
            sam2_checkpoint,
            device=device,
            apply_postprocessing=False
        )

        sam_image(
            sam2=sam2,
            render_images_path=self.render_asset,
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

        save_gpt_input(self.render_asset)
    
    def Pipeline(self) -> None:
        self.asset_visualiser()
        self.asset_segmentation()
    
    def __repr__(self):
        return f"AssetsMaterialPipeline(GLB_folder_path={self.path}, config={self.config})"

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    asset = AssetsMaterialPipeline(GLB_folder_path="test_asset", config=config)
    asset.Pipeline()

if __name__ == "__main__":
    main()