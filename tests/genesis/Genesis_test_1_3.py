import genesis as gs

# import open3d

# mesh = open3d.io.read_triangle_mesh("Tree.glb")
# mesh = mesh.remove_non_manifold_edges()
# open3d.io.write_triangle_mesh("Tree/Tree_repaired.obj", mesh)



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
    fem_options=gs.options.FEMOptions(
        dt=3e-3,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
# franka = scene.add_entity(
#     gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml',
#                 ),
# )
# go = scene.add_entity(
#     gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf",
#                    pos=(0.0, 0.0, 4.0)),
# )

asset1 = scene.add_entity(
    # gs.morphs.Mesh(file='./Tree.glb', 
    #         pos=(0.0, 0.0, 0.5),
    #         euler=(90, 0, 0),
    #         scale=0.6,), 
    # gs.morphs.Sphere(pos=(0.0, 0.0, 0.4),
    #               radius=0.3,
    #               euler=(30, 30, 30),),
    # gs.morphs.Box(pos=(0.0, 0.0, 0.4),
    #               size=(0.2, 0.2, 0.2),
    #               euler=(30, 30, 30),),
    gs.morphs.Mesh(file='meshes/tank.obj', 
            pos=(0.0, 0.0, 5),
            euler=(90, 0, 0),
            scale=1,), 
    material=gs.materials.FEM.Elastic(),
    surface=gs.surfaces.Default(
            vis_mode="visual",
    ),
)

# tank = scene.add_entity(
#     gs.morphs.Mesh(
#         file="meshes/tank.obj",
#         convexify=False,
#         scale=5.0,
#         fixed=True,
#         euler=(90, 0, 0),
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

for i in range(100):
    scene.step()
    cam.set_pose(
        pos    = (10 * np.sin(i / 240), 10 * np.cos(i / 240), 10),
        lookat = (0, 0, 0.5),
    )
    cam.render()
cam.stop_recording(save_to_filename='Duck_FEM.mp4', fps=60)