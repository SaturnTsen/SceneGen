import genesis as gs
gs.init(backend=gs.cuda)

dt = 2e-2
scene = gs.Scene(
    show_viewer = False,
    sim_options=gs.options.SimOptions(
        substeps=20000,
        dt = dt,
    ),
    vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, 0, 0.8),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
    mpm_options=gs.options.MPMOptions(
        dt = dt,
        lower_bound=(-0.2, -0.2, -0.2),
        upper_bound=(0.2, 0.2, 1),
        # enable_CPIC=True,
        grid_density=128,
        particle_size=0.001,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

asset1 = scene.add_entity(
    gs.morphs.Mesh(file='assets/sub_meshes/Bottle_Circle.001.obj', 
        pos=(0.0, 0.0, 0.15),
        euler=(90, 0, 180),
        scale=0.25,
        # convexify=False,
        # coacd_options = gs.options.CoacdOptions(),
        ), 
    material=gs.materials.MPM.Elastic(
        E=2e8,
        rho=920,
        nu=0.25,
        sampler="random",
        ),
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

for i in range(50):
    scene.step()
    cam.set_pose(
        pos    = (1 * np.sin(i / 240), 1 * np.cos(i / 240), 0.5),
        lookat = (0, 0, 0.2),
    )
    cam.render()
cam.stop_recording(save_to_filename=f'./videos/Bottle_MPM_Soft_c.mp4', fps=50)