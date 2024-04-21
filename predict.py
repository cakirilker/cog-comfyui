import os
import shutil
import tarfile
import zipfile
import mimetypes
from PIL import Image
from typing import List
from cog import BasePredictor, Input, Path
from helpers.comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"

mimetypes.add_type("image/webp", ".webp")

with open("examples/api_workflows/sdxl_simple_example.json", "r") as file:
    EXAMPLE_WORKFLOW_JSON = file.read()

source_path = "./checkpoints"
destination_path = "./ComfyUI"

class Predictor(BasePredictor):
    def move_files(self, src, dst):
        try:
            source_checkpoints = os.path.join(src)
            target_checkpoints = os.path.join(dst, 'models/checkpoints')
    
            os.makedirs(target_checkpoints, exist_ok=True)
            files = os.listdir(source_checkpoints)
            for file in files:
                if file.endswith('.safetensors'):
                    src_path = os.path.join(source_checkpoints, file)
                    dst_path = os.path.join(target_checkpoints, file)
                    print(f"⏳ Moving {src_path} to {dst_path}")
                    shutil.move(src_path, dst_path)

            # Move files in upscale_models subdirectory
            source_upscale = os.path.join(src, 'upscale_models')
            target_upscale = os.path.join(dst, 'models/upscale_models')
            os.makedirs(target_upscale, exist_ok=True)
            upscale_files = os.listdir(source_upscale)
            for file in upscale_files:
                src_path = os.path.join(source_upscale, file)
                dst_path = os.path.join(target_upscale, file)
                print(f"⏳ Moving Upscale Model {src_path} to {dst_path}")
                shutil.move(src_path, dst_path)

            # Special case for BRIA-model.pth
            bria_src_path = os.path.join(src, 'BRIA-model.pth')
            bria_dst_path = os.path.join(dst, 'custom_nodes/ComfyUI-BRIA_AI-RMBG/RMBG-1.4/model.pth')
            os.makedirs(os.path.dirname(bria_dst_path), exist_ok=True)
            if os.path.exists(bria_src_path):
                print(f"⏳ Moving RMBG Model from {bria_src_path} to {bria_dst_path}")
                shutil.move(bria_src_path, bria_dst_path)
    
            print("Files have been moved successfully!")
            
        except Exception as e:
            print(f"Error occurred while copying file: {e}")

    def setup(self):
        self.move_files(source_path, destination_path)
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def cleanup(self):
        self.comfyUI.clear_queue()
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def handle_input_file(self, input_file: Path):
        file_extension = os.path.splitext(input_file)[1].lower()
        if file_extension == ".tar":
            with tarfile.open(input_file, "r") as tar:
                tar.extractall(INPUT_DIR)
        elif file_extension == ".zip":
            with zipfile.ZipFile(input_file, "r") as zip_ref:
                zip_ref.extractall(INPUT_DIR)
        elif file_extension in [".jpg", ".jpeg", ".png", ".webp"]:
            shutil.copy(input_file, os.path.join(INPUT_DIR, f"input{file_extension}"))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        print("====================================")
        print(f"Inputs uploaded to {INPUT_DIR}:")
        self.log_and_collect_files(INPUT_DIR)
        print("====================================")

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def predict(
        self,
        workflow_json: str = Input(
            description="Your ComfyUI workflow as JSON. You must use the API version of your workflow. Get it from ComfyUI using ‘Save (API format)’. Instructions here: https://github.com/fofr/cog-comfyui",
            default="",
        ),
        input_file: Path = Input(
            description="Input image, tar or zip file. Read guidance on workflows and input files here: https://github.com/fofr/cog-comfyui. Alternatively, you can replace inputs with URLs in your JSON workflow and the model will download them.",
            default=None,
        ),
        return_temp_files: bool = Input(
            description="Return any temporary files, such as preprocessed controlnet images. Useful for debugging.",
            default=False,
        ),
        optimise_output_images: bool = Input(
            description="Optimise output images by using webp",
            default=True,
        ),
        optimise_output_images_quality: int = Input(
            description="Quality of the output images, from 0 to 100",
            default=80,
        ),
        randomise_seeds: bool = Input(
            description="Automatically randomise seeds (seed, noise_seed, rand_seed)",
            default=True,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        if input_file:
            self.handle_input_file(input_file)

        # TODO: Record the previous models loaded
        # If different, run /free to free up models and memory

        wf = self.comfyUI.load_workflow(workflow_json or EXAMPLE_WORKFLOW_JSON)

        if randomise_seeds:
            self.comfyUI.randomise_seeds(wf)

        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files = []
        output_directories = [OUTPUT_DIR]
        if return_temp_files:
            output_directories.append(COMFYUI_TEMP_OUTPUT_DIR)

        for directory in output_directories:
            print(f"Contents of {directory}:")
            files.extend(self.log_and_collect_files(directory))

        if optimise_output_images:
            optimised_files = []
            for file in files:
                if file.is_file() and file.suffix in [".jpg", ".jpeg", ".png"]:
                    image = Image.open(file)
                    optimised_file_path = file.with_suffix(".webp")
                    image.save(
                        optimised_file_path,
                        quality=optimise_output_images_quality,
                        optimize=True,
                    )
                    optimised_files.append(optimised_file_path)
                else:
                    optimised_files.append(file)

            files = optimised_files

        return files
