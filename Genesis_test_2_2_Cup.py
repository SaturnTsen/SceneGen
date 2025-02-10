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
        lower_bound=(-0.4, -0.4, -0.2),
        upper_bound=(0.4, 0.4, 0.8),
        # enable_CPIC=True,
        grid_density=128,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

asset1 = scene.add_entity(
    gs.morphs.Mesh(file='assets/Cup.obj', 
        pos=(0.0, 0.0, 0.1),
        euler=(90, 0, 180),
        scale=0.025,
        convexify=False,
        coacd_options = gs.options.CoacdOptions(),), 
    material=gs.materials.MPM.Elastic(
        E=7e10,
        rho=2500,
        nu=0.22,
        # sampler="random",
        ),
    surface=gs.surfaces.Default(
            vis_mode="visual",
    ),
)

# asset2 = scene.add_entity(
#     gs.morphs.Mesh(file='assets/Tissue.glb', 
#         pos=(0.05, 0.05, 0.15),
#         euler=(90, 0, 180),
#         scale=0.2,
#         convexify=False,
#         coacd_options = gs.options.CoacdOptions(),), 
#     material=gs.materials.MPM.ElastoPlastic(
#         E=2e8,
#         rho=50,
#         nu=0.3,
#         ),
#     surface=gs.surfaces.Default(
#             vis_mode="visual",
#     ),
# )

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

for i in range(10):
    scene.step()
    cam.set_pose(
        pos    = (1 * np.sin(i / 240), 1 * np.cos(i / 240), 0.5),
        lookat = (0, 0, 0.2),
    )
    cam.render()
cam.stop_recording(save_to_filename=f'./videos/Cup_MPM_pbs.mp4', fps=50)