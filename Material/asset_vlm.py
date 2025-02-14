import os
from utils.vlm_utils import Qwen, GPT4V
from tqdm import tqdm

def query_vlm(
        render_images_path: str,
        vlm_type: str = "Qwen",
        vlm_api_key: str = None,
        vlm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-vl-max-latest",
    ):
    
    vlm = None
    if vlm_type == 'Qwen':
        assert(vlm_api_key is not None), "API key is required for Qwen VLM"
        vlm = Qwen(
            api_key=vlm_api_key,
            base_url=vlm_base_url,
            model=model,
        )
        print()
    
    elif vlm_type == 'GPT4V':
        assert(vlm_api_key is not None), "API key is required for GPT4V VLM"
        vlm = GPT4V(
            api_key=vlm_api_key,
            model=model,
        )
    
    assert(vlm is not None), "VLM type not recognized and VLM needed to be provided"


    for idx, asset in tqdm(enumerate(os.listdir(render_images_path)), desc="VLM Processing Assets"):
        vlm_input_asset_path = os.path.join(render_images_path, asset, 'gpt_input')
        view_list = sorted(os.listdir(vlm_input_asset_path))
        image_files = []
        for view in view_list:
            view_path = os.path.join(vlm_input_asset_path, view)
            for image in os.listdir(view_path):
                image_path = os.path.join(view_path, image)
                image_files.append(image_path)
        
        material_list = "wood, metal, plastic, glass, fabric, foam, food, ceramic, paper, leather"
        material_list = material_list.split(", ")
        material_library = "{" + ", ".join(material_list) + "}"
        prompt = f"""Provided a picture. The first image is the front view of the object (Asset Front View), 
        the second image is the original picture of the object (Original Image), 
        and the third image is a partial segmentation diagram (Mask Overlay), mask is in red. 
        The last image is a partial of the object. 
        
        Based on the image, firstly provide a brief caption of the part. 
        Secondly, describe what the part is made of (provide the major one). 
        Finally, we combine what the object is and the material of the object to predict the hardness of the part. 
        Choose whether to use Shore A hardness or Shore D hardness depending on the material. 
        You may provide a range of values for hardness instead of a single value. 

        Format Requirement:
        You must provide your answer as a (brief caption of the part, material of the part, hardness, Shore A/D) pair. Do not include any other text in your answer, as it will be parsed by a code script later. 
        common material library: {material_library}. 
        Your answer must look like: caption, material, hardness low-high, <Shore A or Shore D>. 
        The material type must be chosen from the above common material library. Make sure to use Shore A or Shore D hardness, not Mohs hardness."""

        output_file = f'{asset}.txt'
        results_file_path = os.path.join(render_images_path, asset, output_file)
        case_msg = ""
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

        with open(results_file_path, 'w') as file:
            for idx, image_file in tqdm(enumerate(image_files), desc=f"Processing images of {asset}", leave=False):
                # try:
                message = str(vlm.query(image_file, prompt))
                # except KeyError as e:
                #     message = "error,-1"
                # except Exception as e:
                #     message = "error,-1"
                write_msg = image_file + "," + message
                case_msg += case_msg
                file.write(f"{write_msg}\n")
                file.flush()

if __name__ == '__main__':
    from api_key import api_key
    render_images_path = "renders"
    vlm_type = "GPT4V"
    vlm_api = api_key
    model = "gpt-4o-mini"
    
    query_vlm(
        render_images_path=render_images_path,
        vlm_type=vlm_type,
        vlm_api_key=vlm_api,
        model=model,
    )