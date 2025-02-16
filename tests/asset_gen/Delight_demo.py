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

from texgen import Hunyuan3DDelightPipeline

pipeline = Hunyuan3DDelightPipeline.from_pretrained("Hunyuan3D-2/ckpts/Hunyuan3D-2")
delighted = pipeline("room.jpg")
delighted.save("delighted.png")