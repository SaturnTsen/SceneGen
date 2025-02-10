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
        particle_size=0.002,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

asset1 = scene.add_entity(
    gs.morphs.Mesh(file='assets/sub_meshes/Bottle_Circle.001.obj', 
        pos=(0.0, 0.0, 0.3),
        euler=(90, 0, 180),
        scale=0.25,
        # convexify=False,
        # coacd_options = gs.options.CoacdOptions(),
        ), 
    material=gs.materials.MPM.ElastoPlastic(
        E = 3e9,
        nu = 0.38,
        rho = 1380,
        # sampler = "regular",
        yield_lower = 2.5e-2,
        yield_higher = 4.5e-3,
        use_von_mises = True,
        von_mises_yield_stress = 55e6,
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

for i in range(200):
    scene.step()
    cam.set_pose(
        pos    = (1 * np.sin(i / 240), 1 * np.cos(i / 240), 0.5),
        lookat = (0, 0, 0.2),
    )
    cam.render()
cam.stop_recording(save_to_filename=f'./videos/Bottle_MPM_Solid.mp4', fps=50)