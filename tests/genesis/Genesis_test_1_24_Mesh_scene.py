import genesis as gs
gs.init(backend=gs.cuda)


scene = gs.Scene(
    show_viewer = False,
    sim_options=gs.options.SimOptions(
        substeps=15,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, 0, 0.8),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

asset1 = scene.add_entity(
    gs.morphs.Mesh(file='room.glb', 
            pos=(0.0, 0.0, 2),
            euler=(90, 0, 180),
            scale=5,
            decimate=False,
            convexify=False,
            fixed=True,
            coacd_options = gs.options.CoacdOptions(),), 
    surface=gs.surfaces.Default(
            vis_mode="visual",
    ),
)

asset2 = scene.add_entity(
    gs.morphs.Mesh(file='Tree.glb', 
            pos=(0.0, 0.0, 5),
            euler=(90, 0, 180),
            scale=0.5,
            decimate=False,
            convexify=False,
            coacd_options = gs.options.CoacdOptions(),), 
    surface=gs.surfaces.Default(
            vis_mode="visual",
    ),
)

cam = scene.add_camera(
    res    = (640, 480),
    pos    = (5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False,
)

scene.build()

# # render rgb, depth, segmentation, and normal
# # rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

cam.start_recording()
import numpy as np

for i in range(400):
    scene.step()
    cam.set_pose(
        pos    = (0, 10, 5),
        lookat = (0, 0, 0.5),
    )
    cam.render()
cam.stop_recording(save_to_filename=f'./videos/Room_mesh.mp4', fps=60)