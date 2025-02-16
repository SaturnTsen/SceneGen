import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
from trellis.utils import render_utils, postprocessing_utils
import importlib.util
import sys
from pathlib import Path

folder_name = "Hunyuan3D-2/hy3dgen"
module_name = "texgen"
file_path = Path(folder_name) / module_name / "__init__.py"

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Load the Hunyuan3D-2.0 delight model
# delight_model = module.Hunyuan3DDelightPipeline.from_pretrained("Hunyuan3D-2/ckpts/Hunyuan3D-2")
delight_model = None

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("../TRELLIS-image-large/")
pipeline.cuda()

# Load an image
images = [
    Image.open("assets/Tissue/1.jpg"),
    Image.open("assets/Tissue/2.jpg"),
    Image.open("assets/Tissue/3.jpg"),
    Image.open("assets/Tissue/4.jpg"),
]

# Run the pipeline
outputs, delight_images = pipeline.run_multi_image(
    images,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 25,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 25,
        "cfg_strength": 3,
    },
    delight_image_model=delight_model,
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

Asset_name = "Tissue"
if not os.path.exists(Asset_name):
    os.makedirs(Asset_name)

video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave(f"{Asset_name}/{Asset_name}_video.mp4", video, fps=30)

# if delight_image is not None:
#     delight_image.save(f"{Asset_name}/{Asset_name}_delight.png")

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export(f"{Asset_name}/{Asset_name}.glb")
os.system(f"mv {Asset_name}/ Demos")

