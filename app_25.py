import os
import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextConfig
from safetensors.torch import load_file
from collections import OrderedDict
import re
import json
import gdown
#import requests # Removed
import subprocess
from urllib.parse import urlparse, unquote
from pathlib import Path
import tempfile
# from tqdm import tqdm  # Removed: not crucial and can break display in gradio.
import psutil
import math
import shutil
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from huggingface_hub import login, HfApi, hf_hub_download
from huggingface_hub.file_download import get_from_cache  # Corrected import
from huggingface_hub.utils import validate_repo_id, HFValidationError
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE  # Import HUGGINGFACE_HUB_CACHE


# ---------------------- DEPENDENCIES ----------------------
def install_dependencies_gradio():
    """Installs the necessary dependencies."""
    try:
        subprocess.run(
            [
                "pip",
                "install",
                "-U",
                "torch",
                "diffusers",
                "transformers",
                "accelerate",
                "safetensors",
                "huggingface_hub",
                "xformers",
            ]
        )
        print("Dependencies installed successfully.")
    except Exception as e:
        print(f"Error installing dependencies: {e}")


# ---------------------- UTILITY FUNCTIONS ----------------------


def increment_filename(filename):
    """Increments the filename to avoid overwriting existing files."""
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base}({counter}){ext}"
        counter += 1
    return filename


# ---------------------- UPLOAD FUNCTION ----------------------
def create_model_repo(api, user, orgs_name, model_name, make_private=False):
    """Creates a Hugging Face model repository."""
    repo_id = (
        f"{orgs_name}/{model_name.strip()}"
        if orgs_name
        else f"{user['name']}/{model_name.strip()}"
    )
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=make_private)
        print(f"Model repo '{repo_id}' created.")
    except HfHubHTTPError:
        print(f"Model repo '{repo_id}' already exists.")
    return repo_id


# ---------------------- MODEL LOADING AND CONVERSION ----------------------
def download_model(model_path_or_url):
    """Downloads a model from a Hugging Face Hub repository."""
    try:
        # Check if it's a valid Hugging Face repo ID (and potentially a file within)
        try:
            validate_repo_id(model_path_or_url)
            # It's a valid repo ID; use hf_hub_download (it handles caching)
            local_path = hf_hub_download(repo_id=model_path_or_url)
            return local_path
        except HFValidationError:
            # Might be a repo ID + filename
            try:
                parts = model_path_or_url.split("/", 1)
                if len(parts) == 2:
                    repo_id, filename = parts
                    validate_repo_id(repo_id)
                    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    return local_path
                else:
                    raise ValueError("Invalid Hugging Face repository format.")
            except HFValidationError:
                raise ValueError(f"Invalid Hugging Face repository ID or path: {model_path_or_url}")

    except Exception as e:
        raise ValueError(f"Error downloading model: {e}")


def load_sdxl_checkpoint(checkpoint_path):
    """Loads an SDXL checkpoint (.ckpt or .safetensors) and returns components."""

    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path, device="cpu")  # Load to CPU
    elif checkpoint_path.endswith(".ckpt"):
        state_dict = torch.load(checkpoint_path, map_location="cpu")[
            "state_dict"
        ]  # Load to CPU, access ["state_dict"]
    else:
        raise ValueError("Unsupported checkpoint format. Must be .safetensors or .ckpt")

    text_encoder1_state = OrderedDict()
    text_encoder2_state = OrderedDict()
    vae_state = OrderedDict()
    unet_state = OrderedDict()

    for key, value in state_dict.items():
        if key.startswith("first_stage_model."):  # VAE
            vae_state[key.replace("first_stage_model.", "")] = value.to(
                torch.float16
            )  # FP16 conversion
        elif key.startswith("condition_model.model.text_encoder."):  # Text Encoder 1
            text_encoder1_state[
                key.replace("condition_model.model.text_encoder.", "")
            ] = value.to(
                torch.float16
            )  # FP16
        elif key.startswith(
            "condition_model.model.text_encoder_2."
        ):  # Text Encoder 2
            text_encoder2_state[
                key.replace("condition_model.model.text_encoder_2.", "")
            ] = value.to(
                torch.float16
            )  # FP16
        elif key.startswith("model.diffusion_model."):  # UNet
            unet_state[key.replace("model.diffusion_model.", "")] = value.to(
                torch.float16
            )  # FP16

    return text_encoder1_state, text_encoder2_state, vae_state, unet_state


def build_diffusers_model(
    text_encoder1_state, text_encoder2_state, vae_state, unet_state, reference_model_path=None
):
    """Builds the Diffusers pipeline components from the loaded state dicts."""

    # Default to SDXL base 1.0 if no reference model is provided
    if not reference_model_path:
        reference_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    # 1. Text Encoders
    config_text_encoder1 = CLIPTextConfig.from_pretrained(
        reference_model_path, subfolder="text_encoder"
    )
    config_text_encoder2 = CLIPTextConfig.from_pretrained(
        reference_model_path, subfolder="text_encoder_2"
    )

    text_encoder1 = CLIPTextModel(config_text_encoder1)
    text_encoder2 = CLIPTextModel(config_text_encoder2)
    text_encoder1.load_state_dict(text_encoder1_state)
    text_encoder2.load_state_dict(text_encoder2_state)
    text_encoder1.to(torch.float16).to("cpu")  # Ensure fp16 and CPU
    text_encoder2.to(torch.float16).to("cpu")

    # 2. VAE
    vae = AutoencoderKL.from_pretrained(reference_model_path, subfolder="vae")
    vae.load_state_dict(vae_state)
    vae.to(torch.float16).to("cpu")

    # 3. UNet
    unet = UNet2DConditionModel.from_pretrained(reference_model_path, subfolder="unet")
    unet.load_state_dict(unet_state)
    unet.to(torch.float16).to("cpu")

    return text_encoder1, text_encoder2, vae, unet


def convert_and_save_sdxl_to_diffusers(
    checkpoint_path_or_url, output_path, reference_model_path
):
    """Converts an SDXL checkpoint to Diffusers format and saves it.

    Args:
        checkpoint_path_or_url:  The path/URL/repo ID of the checkpoint.
    """

    # Download the model if necessary (handles URLs, repo IDs, and local paths)
    checkpoint_path = download_model(checkpoint_path_or_url)

    text_encoder1_state, text_encoder2_state, vae_state, unet_state = (
        load_sdxl_checkpoint(checkpoint_path)
    )
    text_encoder1, text_encoder2, vae, unet = build_diffusers_model(
        text_encoder1_state,
        text_encoder2_state,
        vae_state,
        unet_state,
        reference_model_path,
    )

    # Load tokenizer and scheduler from the reference model
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        reference_model_path,
        text_encoder=text_encoder1,
        text_encoder_2=text_encoder2,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16,
    )
    pipeline.to("cpu")
    pipeline.save_pretrained(output_path)
    print(f"Model saved as Diffusers format: {output_path}")


# ---------------------- UPLOAD FUNCTION ----------------------
def upload_to_huggingface(model_path, hf_token, orgs_name, model_name, make_private):
    """Uploads a model to the Hugging Face Hub."""
    login(hf_token, add_to_git_credential=True)
    api = HfApi()
    user = api.whoami(hf_token)
    model_repo = create_model_repo(api, user, orgs_name, model_name, make_private)
    api.upload_folder(folder_path=model_path, repo_id=model_repo)
    print(f"Model uploaded to: https://huggingface.co/{model_repo}")


# ---------------------- GRADIO INTERFACE ----------------------
def main(model_to_load, reference_model, output_path, hf_token, orgs_name, model_name, make_private):
    """Main function: SDXL checkpoint to Diffusers, always fp16."""

    try:
        convert_and_save_sdxl_to_diffusers(model_to_load, output_path, reference_model)
        upload_to_huggingface(output_path, hf_token, orgs_name, model_name, make_private)
        return "Conversion and upload completed successfully!"
    except Exception as e:
        return f"An error occurred: {e}"  # Return the error message


css = """
#main-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    justify-content: space-between;
    font-family: 'Arial', sans-serif;
    font-size: 16px;
    color: #333;
}
#convert-button {
    margin-top: auto;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
    # 🎨 SDXL Model Converter
    Convert SDXL checkpoints to Diffusers format (FP16, CPU-only).

    ### 📥 Input Sources Supported:
    - Hugging Face model repositories (e.g., 'my-org/my-model' or 'my-org/my-model/file.safetensors')

    ### ℹ️ Important Notes:
    - This tool runs on **CPU**, conversion might be slower than on GPU.
    - For Hugging Face uploads, you need a **WRITE** token (not a read token).
    - Get your HF token here: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

    ### 💾 Memory Usage:
    - This space is configured for **FP16** precision to reduce memory usage.
    - Close other applications during conversion.
    - For large models, ensure you have at least 16GB of RAM.

    ### 💻 Source Code:
    - [GitHub Repository](https://github.com/Ktiseos-Nyx/Gradio-SDXL-Diffusers)

    ### 🙏 Support:
    - If you're interested in funding more projects: [Ko-fi](https://ko-fi.com/duskfallcrew)
    """
    )

    with gr.Column(elem_id="main-container"):  # Use a Column for layout
        model_to_load = gr.Textbox(
            label="SDXL Checkpoint (HF Repo)",  # More specific label
            placeholder="Hugging Face Repo ID (e.g., my-org/my-model or my-org/my-model/file.safetensors)",
        )
        reference_model = gr.Textbox(
            label="Reference Diffusers Model (Optional)",
            placeholder="e.g., stabilityai/stable-diffusion-xl-base-1.0 (Leave blank for default)",
        )
        output_path = gr.Textbox(
            label="Output Path (Diffusers Format)", value="output"
        )  # Default changed to "output"
        hf_token = gr.Textbox(
            label="Hugging Face Token", placeholder="Your Hugging Face write token"
        )
        orgs_name = gr.Textbox(
            label="Organization Name (Optional)", placeholder="Your organization name"
        )
        model_name = gr.Textbox(
            label="Model Name", placeholder="The name of your model on Hugging Face"
        )
        make_private = gr.Checkbox(label="Make Repository Private", value=False)

        convert_button = gr.Button("Convert and Upload", elem_id="convert-button")
        output = gr.Markdown()

    convert_button.click(
        fn=main,
        inputs=[
            model_to_load,
            reference_model,
            output_path,
            hf_token,
            orgs_name,
            model_name,
            make_private,
        ],
        outputs=output,
    )

demo.launch()
