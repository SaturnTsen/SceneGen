import os
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

def extract_submeshes(file_path):
    output_dir = "assets/sub_meshes"
    os.makedirs(output_dir, exist_ok=True)
    
    if file_path.lower().endswith('.obj'):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Global vertex, texture, normal definitions (note: indices start from 1)
        vertices = [line for line in lines if line.startswith('v ')]
        textures = [line for line in lines if line.startswith('vt ')]
        normals  = [line for line in lines if line.startswith('vn ')]
        
        # Divide by submesh based on "o" or "g" tags
        submesh_data = {}
        current_name = None
        current_lines = []
        for line in lines:
            if line.startswith("o ") or line.startswith("g "):
                if current_name is not None:
                    submesh_data[current_name] = current_lines
                parts = line.strip().split()
                current_name = parts[1] if len(parts) > 1 else "unnamed"
                current_lines = [line]
            else:
                if current_name is not None:
                    current_lines.append(line)
        if current_name is not None:
            submesh_data[current_name] = current_lines

        # Process each submesh: extract used vertices, textures, normals and reindex, normalize vertices to unit space and center the origin
        for name, sub_lines in submesh_data.items():
            used_v = set()
            used_vt = set()
            used_vn = set()
            face_lines = []
            for line in sub_lines:
                if line.startswith("f "):
                    face_lines.append(line)
                    parts = line.strip().split()[1:]
                    for part in parts:
                        indices = part.split('/')
                        if indices[0]:
                            used_v.add(int(indices[0]))
                        if len(indices) > 1 and indices[1]:
                            used_vt.add(int(indices[1]))
                        if len(indices) > 2 and indices[2]:
                            used_vn.add(int(indices[2]))
            
            # Extract used vertex coordinates
            sub_vertices_coords = {}
            for idx in sorted(used_v):
                try:
                    v_line = vertices[idx - 1].strip()  # Format: "v x y z"
                    parts = v_line.split()
                    x, y, z = map(float, parts[1:4])
                    sub_vertices_coords[idx] = (x, y, z)
                except IndexError:
                    pass
            
            # Calculate bounding box and longest edge
            xs = [coord[0] for coord in sub_vertices_coords.values()]
            ys = [coord[1] for coord in sub_vertices_coords.values()]
            zs = [coord[2] for coord in sub_vertices_coords.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            min_z, max_z = min(zs), max(zs)
            scale = max(max_x - min_x, max_y - min_y, max_z - min_z)
            
            # Create normalized vertices: translate and normalize, then move origin to center (-0.5 offset)
            normalized_vertices = {}
            for idx, (x, y, z) in sub_vertices_coords.items():
                x_norm = (x - min_x) / scale - 0.5
                y_norm = (y - min_y) / scale - 0.5
                z_norm = (z - min_z) / scale - 0.5
                normalized_vertices[idx] = f"v {x_norm} {y_norm} {z_norm}\n"
            
            # Create old index -> new index mapping (sorted in ascending order)
            v_map  = {old: new for new, old in enumerate(sorted(used_v), start=1)}
            vt_map = {old: new for new, old in enumerate(sorted(used_vt), start=1)}
            vn_map = {old: new for new, old in enumerate(sorted(used_vn), start=1)}

            out_file = os.path.join(output_dir, f"{name}.obj")
            with open(out_file, 'w', encoding='utf-8') as fout:
                # Write material library declaration if present in the original file
                for line in lines:
                    if line.startswith("mtllib"):
                        fout.write(line)
                # Output normalized vertices
                for idx in sorted(used_v):
                    try:
                        fout.write(normalized_vertices[idx])
                    except IndexError:
                        pass
                for idx in sorted(used_vt):
                    try:
                        fout.write(textures[idx - 1])
                    except IndexError:
                        pass
                for idx in sorted(used_vn):
                    try:
                        fout.write(normals[idx - 1])
                    except IndexError:
                        pass
                fout.write(f"o {name}\n")
                for line in face_lines:
                    parts = line.strip().split()
                    new_face = "f"
                    for part in parts[1:]:
                        indices = part.split('/')
                        v_new = vt_new = vn_new = ""
                        if indices[0]:
                            v_new = str(v_map[int(indices[0])])
                        if len(indices) > 1 and indices[1]:
                            vt_new = str(vt_map[int(indices[1])])
                        if len(indices) > 2 and indices[2]:
                            vn_new = str(vn_map[int(indices[2])])
                        new_vertex = v_new
                        if vt_new or vn_new:
                            new_vertex += "/" + vt_new
                        if vn_new:
                            new_vertex += "/" + vn_new
                        new_face += " " + new_vertex
                    fout.write(new_face + "\n")
            print(f"Saved submesh {name} to file: {out_file}")
    
    elif file_path.lower().endswith('.glb'):
        # Use trimesh to load glb file, normalize each submesh separately and export
        scene = trimesh.load(file_path, force='scene')
        for name, geom in scene.geometry.items():
            vertices = geom.vertices
            faces = geom.faces
            # Calculate bounding box and longest edge
            min_bounds = vertices.min(axis=0)
            max_bounds = vertices.max(axis=0)
            scale = (max_bounds - min_bounds).max()
            # Normalize vertices, normalize to [0,1] then translate to center the origin
            normalized_vertices = (vertices - min_bounds) / scale - 0.5
            # Create new mesh
            new_mesh = trimesh.Trimesh(vertices=normalized_vertices, faces=faces)
            out_file = os.path.join(output_dir, f"{name}.obj")
            new_mesh.export(out_file)
            print(f"Saved submesh {name} to file: {out_file}")
    else:
        print("Unsupported file format")

def merge_submeshes(file_path):
    os.makedirs("assets/merge_meshes", exist_ok=True)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    mtllib_lines = []
    vertex_lines = []
    texture_lines = []
    normal_lines = []
    face_lines = []
    
    for line in lines:
        if line.startswith("mtllib"):
            mtllib_lines.append(line)
        elif line.startswith("v "):
            vertex_lines.append(line)
        elif line.startswith("vt "):
            texture_lines.append(line)
        elif line.startswith("vn "):
            normal_lines.append(line)
        elif line.startswith("f "):
            face_lines.append(line)
    
    # Parse vertex coordinates to calculate bounding box
    vertices_coords = []
    for line in vertex_lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        x, y, z = map(float, parts[1:4])
        vertices_coords.append((x, y, z))
        
    if not vertices_coords:
        print("No vertex data found")
        return
    
    xs = [coord[0] for coord in vertices_coords]
    ys = [coord[1] for coord in vertices_coords]
    zs = [coord[2] for coord in vertices_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    scale = max(max_x - min_x, max_y - min_y, max_z - min_z)
    
    # Normalize vertices: translate and normalize, then center the origin (-0.5 offset)
    normalized_vertex_lines = []
    for line in vertex_lines:
        parts = line.strip().split()
        if len(parts) < 4:
            normalized_vertex_lines.append(line)
        else:
            x, y, z = map(float, parts[1:4])
            x_norm = (x - min_x) / scale - 0.5
            y_norm = (y - min_y) / scale - 0.5
            z_norm = (z - min_z) / scale - 0.5
            normalized_vertex_lines.append(f"v {x_norm} {y_norm} {z_norm}\n")
            
    merged_file = os.path.join("assets/merge_meshes", f"{os.path.basename(file_path).split('.')[0]}_merge.obj")
    with open(merged_file, 'w', encoding='utf-8') as fout:
        # Output material library declaration (if present)
        for line in mtllib_lines:
            fout.write(line)
        # Output normalized vertex data
        for v_line in normalized_vertex_lines:
            fout.write(v_line)
        # Output texture and normal data (keep original order)
        for vt_line in texture_lines:
            fout.write(vt_line)
        for vn_line in normal_lines:
            fout.write(vn_line)
        # Output merged object name
        fout.write("o merged\n")
        # Output all face data (merge all submeshes together)
        for f_line in face_lines:
            fout.write(f_line)
    print(f"Merged file saved as: {merged_file}")

def glb_asset_rotator(file_path,
                      euler_x: float = 90.0,
                      euler_y: float = 0.0,
                      euler_z: float = 180.0,
                      replace_org_file: bool = False):
    # Load the GLB file
    scene = trimesh.load(file_path, force='scene')

    # Create a rotation matrix from Euler angles
    rotation = R.from_euler('xyz', [euler_x, euler_y, euler_z], degrees=True).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    
    for name, geom in scene.geometry.items():
        geom.apply_transform(transform)

    # Save the rotated scene back to the file
    if not replace_org_file:
        output_file = os.path.splitext(file_path)[0] + '_rotated.glb'
        scene.export(output_file)
        # print(f"Rotated asset saved to file: {output_file}")
        return output_file
    else:
        output_file = file_path
        scene.export(output_file)
        # print(f"Rotated asset saved to file: {output_file}")
        return output_file



if __name__ == "__main__":
    # You can pass .obj or .glb file paths as needed
    # extract_submeshes("assets/Bottle.obj")
    # For glb files, you can call the split or merge submesh functions separately
    # extract_submeshes("assets/Tin.glb")
    # merge_submeshes("assets/soda_can.obj")
    glb_asset_rotator("assets/Tree.glb")