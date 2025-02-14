import os
import argparse
import trimesh
import numpy as np
from pyglet import gl
from tqdm import tqdm
from trimesh.transformations import transform_points
from utils.asset_processor import glb_asset_rotator
from matplotlib import pyplot as plt
os.environ['DISPLAY'] = ':1'


def render_views(
        glb_path: str,
        out_dir: str = 'renders',
        azimuth_angles: list = [0, 120, 240],
        elevation_angles: list = [0, 60, -60],
        trillis_asset: bool = True,
        replace_org_file: bool = False,
        fov_deg: float = 60,
        resolution: tuple = (800, 600),
        mark: bool = False,
        ):
    """
    Render images from various azimuth and elevation angles.
    
    Parameters:
        glb_path: Path to the GLB asset.
        out_dir: Output directory for rendered images.
        azimuth_angles: List of azimuth angles (in degrees).
        elevation_angles: List of elevation angles (in degrees).
        trillis_asset: Whether the asset is a Trillis asset.
        replace_org_file: Whether to replace the original file with the rotated one.
        fov_deg: Field of view angle (in degrees).
        resolution: Image resolution (width, height).
        mark: Whether to mark a point and axes in the image.
    
    Returns:
        Transforms: List of camera transforms.
        Ks: List of intrinsic matrices.
    """
    out_dir = out_dir + '/' + os.path.basename(glb_path).split('.')[0] + '/' + "images"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB asset not found at {glb_path}")

    if trillis_asset:

        # if the asset is a Trillis asset, rotate it to the correct orientation
        # If you what to replace the original file with the rotated one, set replace_org_file=True
        # NOTE: This will overwrite the original file

        # print("Rotating asset to correct orientation...")
        if os.path.exists(glb_path.replace('.glb', '_rotated.glb')):
            glb_path = glb_path.replace('.glb', '_rotated.glb')
        else:
            glb_path = glb_asset_rotator(glb_path, replace_org_file=replace_org_file)
    
    # Load the GLB asset
    scene = trimesh.load(glb_path)
    
    # Compute the center and size of the object's bounding box
    center = scene.bounding_box.centroid
    size = np.max(scene.bounding_box.extents)
    # Set camera distance (adjustable, relative to object size)
    distance = size * 2.0
    # Get all vertices in the scene
    points = np.concatenate([mesh.vertices for mesh in scene.geometry.values()], axis=0)
    
    Transforms = []
    Ks = []
    render_index = 0

    if mark:
        # Add axes and center marker to the scene
        axes = trimesh.creation.axis(origin_size=0.05, axis_length=0.5)
        scene.add_geometry(axes)
        center_marker = trimesh.creation.icosphere(
            radius=0.01, 
            subdivisions=2,
            face_colors=[1, 0, 0],
            )
        center_marker.apply_translation(np.array([0.5,0.5,0.5]))
        scene.add_geometry(center_marker)

    # Loop through each combination of azimuth and elevation angles
    for _, elev in tqdm(enumerate(elevation_angles), desc="Elevation Angles", leave=False):
        for _, azim in tqdm(enumerate(azimuth_angles), desc="Azimuth Angles", leave=False):
            # Set the camera transform for the current view
            # NOTE: The basic euler_x was changed to 90-euler_x 
            # and the euler_z was changed to 180-euler_z 
            # to match the camera orientation in Trillis
            rotation_matrix = trimesh.transformations.quaternion_matrix(
                    trimesh.transformations.quaternion_from_euler(
                        np.deg2rad(90 - elev),
                        np.deg2rad(0),
                        np.deg2rad(180 - azim),
                        axes='sxyz'
                        )
                    )
            camera_transform= trimesh.scene.cameras.look_at(
                points=points,
                fov=fov_deg,
                distance=distance,
                rotation=rotation_matrix,
                center=center,
            )
            scene.camera_transform = camera_transform

            # Render the image in offscreen mode (avoid display dependency)
            render_file = os.path.join(out_dir, f'render_{render_index}.png')
            render_index += 1
            window_conf = gl.Config(double_buffer=True, depth_size=24)
            png = scene.save_image(visible=False, resolution=resolution, window_conf=window_conf)
            with open(render_file, 'wb') as f:
                f.write(png)
            
            # Save the camera transform and intrinsic matrix
            camera = scene.camera
            transform = scene.camera_transform
            K = camera.K
            Transforms.append(transform)
            Ks.append(K)

            if mark:
                # Project a mark point to the image and plot it
                mark_point = np.array([[0.5, 0.5, 0.5]])
                transformed = transform_points(mark_point, np.linalg.inv(transform))
                projected = transformed @ K.T
                # Calculate the homogeneous coordinate
                xy = projected[:, :2] / projected[:, 2:]
                xy[:, 0] = scene.camera.resolution[0] - 1 - xy[:, 0]

                image = plt.imread(render_file)
                plt.imshow(image)
                plt.scatter(xy[:, 0], xy[:, 1], c='r', s=20)
                render_file = os.path.join(out_dir, f'render_azim{azim}_elev{elev}_marked.png')
                plt.savefig(render_file)
                plt.close()

            # print(f"Rendered view at azimuth {azim}° and elevation {elev}° saved to {render_file}")
    return Transforms, Ks

if __name__ == '__main__':
    # Example: Render multiple views of the asset from various azimuth and elevation angles
    args = argparse.ArgumentParser()
    args.add_argument("--glb_path", type=str, default="assets/Tin.glb")
    args.add_argument("--out_dir", type=str, default="renders")
    args.add_argument("--azimuth_angles", type=list, default=[0, 120, 240])
    args.add_argument("--elevation_angles", type=list, default=[0, 60, -60])
    args.add_argument("--trillis_asset", type=bool, default=True)
    args.add_argument("--replace_org_file", type=bool, default=False)
    args.add_argument("--fov_deg", type=float, default=60)
    args.add_argument("--resolution", type=tuple, default=(800, 600))
    args.add_argument("--mark", type=bool, default=False)
    args = args.parse_args()


    glb_path = "assets/Tree.glb"
    render_views(
        glb_path=args.glb_path,
        out_dir=args.out_dir,
        azimuth_angles=args.azimuth_angles,
        elevation_angles=args.elevation_angles,
        trillis_asset=args.trillis_asset,
        replace_org_file=args.replace_org_file,
        fov_deg=args.fov_deg,
        resolution=args.resolution,
        mark=args.mark,
    )