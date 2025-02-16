import genesis as gs
gs.init(backend=gs.cuda)

dt = 2e-2
scene = gs.Scene(
    show_viewer = False,
    sim_options=gs.options.SimOptions(
        substeps=20,
        dt = dt,
    ),
    # vis_options=gs.options.VisOptions(
    #         visualize_mpm_boundary=True,
    # ),
    # mpm_options=gs.options.MPMOptions(
    #     dt = dt,
    #     lower_bound=(-0.2, -0.2, -0.2),
    #     upper_bound=(0.2, 0.2, 1),
    #     # enable_CPIC=True,
    #     grid_density=128,
    #     # particle_size=0.003,
    # ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

asset1 = scene.add_entity(
    gs.morphs.Mesh(file='assets/Tree_rotated.glb', 
        pos=(-0.075, 0.075, 0.15),
        euler=(0, 0, 0),
        scale=0.3,
        # convexify=False,
        # coacd_options = gs.options.CoacdOptions(),
        ), 
    # material=gs.materials.MPM.Elastic(
    #     E = 7e10,
    #     nu = 0.33,
    #     rho = 2700,
    #     # sampler = "regular",
    #     ),
    surface=gs.surfaces.Default(
            vis_mode="visual",
    ),
)

asset2 = scene.add_entity(
    gs.morphs.Mesh(file='meshes/sphere.obj',
                   pos=(0, 0, 1),
                   scale=0.15),
    material=gs.materials.Rigid(
        rho=300000,
        friction=4.0,
    ),
    surface=gs.surfaces.Default(
            vis_mode="visual",
            opacity=0.5,
    ),
)

asset2 = scene.add_entity(
    gs.morphs.Mesh(file='meshes/sphere.obj',
                   pos=(0, 0, 4),
                   scale=0.15),
    material=gs.materials.Rigid(
        rho=300000,
        friction=4.0,
    ),
    surface=gs.surfaces.Default(
            vis_mode="visual",
            opacity=0.5,
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
        pos    = (1 * np.sin(i / 240), 1 * np.cos(i / 240), 0.5),
        lookat = (0, 0, 0.2),
    )
    cam.render()
cam.stop_recording(save_to_filename=f'./videos/Rotation_test.mp4', fps=50)