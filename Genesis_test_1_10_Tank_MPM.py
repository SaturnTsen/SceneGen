import genesis as gs
gs.init(backend=gs.cuda)

for iters in range(1):

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
        mpm_options=gs.options.MPMOptions(
            dt = 3e-3,
            lower_bound=(-1.0, -1.0, -0.2),
            upper_bound=(1.0, 1.0, 1.8),
        ),
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    asset1 = scene.add_entity(
        # gs.morphs.Box(pos=(0.0, 0.0, 0.4),
        #               size=(0.2, 0.2, 0.2),
        #               euler=(30, 30, 30),),
        gs.morphs.FileMorph(file='sample.ply', 
                pos=(0.0, 0.0, 0.5),
                euler=(90, 0, 0),
                scale=5,), 
        # material=[gs.materials.MPM.Elastic(
        #     E=1e4,
        #     rho=1190,
        #     ),gs.materials.MPM.Elastic(
        #     E=9e3,
        #     rho=1190,
        #     ),],
        material=gs.materials.MPM.Elastic(
            E=1e3,
            rho=1190,
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

    for i in range(400):
        scene.step()
        cam.set_pose(
            pos    = (2 * np.sin(i / 240), 2 * np.cos(i / 240), 2),
            lookat = (0, 0, 0.5),
        )
        cam.render()
    cam.stop_recording(save_to_filename=f'./videos/Tank_MPM_Material.mp4', fps=60)