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
import requests
import subprocess
from urllib.parse import urlparse, unquote
from pathlib import Path
import tempfile
#from tqdm import tqdm # Removed as not crucial and can break display in gradio.
import psutil
import math
import shutil
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from huggingface_hub import login, HfApi, hf_hub_download  # Import hf_hub_download
from huggingface_hub.utils import validate_repo_id, HFValidationError
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub import HfApi, hf_hub_download, cached_download, get_from_cache  # Import cached_download and get_from_cache
from huggingface_hub.utils import validate_repo_id, HFValidationError
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

# ---------------------- DEPENDENCIES ----------------------
def install_dependencies_gradio():
    """Installs the necessary dependencies."""
    try:
        subprocess.run(["pip", "install", "-U", "torch", "diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub", "xformers"])
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
    repo_id = f"{orgs_name}/{model_name.strip()}" if orgs_name else f"{user['name']}/{model_name.strip()}"
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=make_private)
        print(f"Model repo '{repo_id}' created.")
    except HfHubHTTPError:
        print(f"Model repo '{repo_id}' already exists.")
    return repo_id

# ---------------------- MODEL LOADING AND CONVERSION ----------------------
def download_model(model_path_or_url):
    """Downloads a model, handling URLs, HF repos, and local paths, caching appropriately."""
    try:
        # 1. Check if it's a valid Hugging Face repo ID (and potentially a file within)
        try:
            validate_repo_id(model_path_or_url)
            # It's a valid repo ID; use hf_hub_download
            local_path = hf_hub_download(repo_id=model_path_or_url)
            return local_path
        except HFValidationError:
            pass  # Not a simple repo ID. Might be repo ID + filename, or a URL.

        # 2. Check if it's a URL
        if model_path_or_url.startswith("http://") or model_path_or_url.startswith("https://"):
            # Check if it's already in the cache
            cache_path = get_from_cache(model_path_or_url)  # Use get_from_cache
            if cache_path is not None:
                return cache_path

            # It's a URL and not in cache: download manually and put into HF cache
            response = requests.get(model_path_or_url, stream=True)
            response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

            # Get filename from URL, or use a hash if we can't determine it
            parsed_url = urlparse(model_path_or_url)
            filename = os.path.basename(unquote(parsed_url.path))
            if not filename:
              filename = hashlib.sha256(model_path_or_url.encode()).hexdigest()

            # Construct the cache path (using HF_HUB_CACHE + "downloads" )
            cache_dir = os.path.join(HUGGINGFACE_HUB_CACHE, "downloads")
            os.makedirs(cache_dir, exist_ok=True) # Ensure the cache directory exists
            local_path = os.path.join(cache_dir, filename)

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return local_path

        # 3. Check if it's a local file
        elif os.path.isfile(model_path_or_url):
            return model_path_or_url

        # 4. Handle Hugging Face repo with a specific file
        else:
            try:
                parts = model_path_or_url.split("/", 1)
                if len(parts) == 2:
                    repo_id, filename = parts
                    validate_repo_id(repo_id)
                    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    return local_path
                else:
                   raise ValueError("Invalid input format.")

            except HFValidationError:
                raise ValueError(f"Invalid model path or URL: {model_path_or_url}")

    except Exception as e:
        raise ValueError(f"Error downloading or accessing model: {e}")


def load_sdxl_checkpoint(checkpoint_path):
    """Loads an SDXL checkpoint (.ckpt or .safetensors) and returns components."""

    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path, device="cpu")
    elif checkpoint_path.endswith(".ckpt"):
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    else:
        raise ValueError("Unsupported checkpoint format. Must be .safetensors or .ckpt")

    text_encoder1_state = OrderedDict()
    text_encoder2_state = OrderedDict()
    vae_state = OrderedDict()
    unet_state = OrderedDict()

    for key, value in state_dict.items():
        if key.startswith("first_stage_model."):  # VAE
            vae_state[key.replace("first_stage_model.", "")] = value.to(torch.float16)
        elif key.startswith("condition_model.model.text_encoder."):  # Text Encoder 1
            text_encoder1_state[key.replace("condition_model.model.text_encoder.", "")] = value.to(torch.float16)
        elif key.startswith("condition_model.model.text_encoder_2."):  # Text Encoder 2
            text_encoder2_state[key.replace("condition_model.model.text_encoder_2.", "")] = value.to(torch.float16)
        elif key.startswith("model.diffusion_model."):  # UNet
            unet_state[key.replace("model.diffusion_model.", "")] = value.to(torch.float16)

    return text_encoder1_state, text_encoder2_state, vae_state, unet_state

def build_diffusers_model(text_encoder1_state, text_encoder2_state, vae_state, unet_state, reference_model_path=None):
    """Builds the Diffusers pipeline components from the loaded state dicts."""

    # Default to SDXL base 1.0 if no reference model is provided
    if not reference_model_path:
        reference_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    # 1. Text Encoders
    config_text_encoder1 = CLIPTextConfig.from_pretrained(reference_model_path, subfolder="text_encoder")
    config_text_encoder2 = CLIPTextConfig.from_pretrained(reference_model_path, subfolder="text_encoder_2")

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



def convert_and_save_sdxl_to_diffusers(checkpoint_path_or_url, output_path, reference_model_path):
    """Converts an SDXL checkpoint to Diffusers format and saves it.
    Args:
        checkpoint_path_or_url:  The path/URL/repo ID of the checkpoint.
    """

    # Download the model if necessary (handles URLs, repo IDs, and local paths)
    checkpoint_path = download_model(checkpoint_path_or_url)

    text_encoder1_state, text_encoder2_state, vae_state, unet_state = load_sdxl_checkpoint(checkpoint_path)
    text_encoder1, text_encoder2, vae, unet = build_diffusers_model(text_encoder1_state, text_encoder2_state, vae_state, unet_state, reference_model_path)


    # Load tokenizer and scheduler from the reference model
    pipeline = StableDiffusionXLPipeline.from_pretrained(reference_model_path,
                                                         text_encoder=text_encoder1,
                                                         text_encoder_2=text_encoder2,
                                                         vae=vae,
                                                         unet=unet,
                                                         torch_dtype=torch.float16,)
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
        return f"An error occurred: {e}" # Return the error message


with gr.Blocks() as demo:
    model_to_load = gr.Textbox(label="SDXL Checkpoint (Path, URL, or HF Repo)", placeholder="Path, URL, or Hugging Face Repo ID (e.g., my-org/my-model or my-org/my-model/file.safetensors)")
    reference_model = gr.Textbox(label="Reference Diffusers Model (Optional)", placeholder="e.g., stabilityai/stable-diffusion-xl-base-1.0 (Leave blank for default)")
    output_path = gr.Textbox(label="Output Path (Diffusers Format)", value="output")  # Default changed to "output"
    hf_token = gr.Textbox(label="Hugging Face Token", placeholder="Your Hugging Face write token")
    orgs_name = gr.Textbox(label="Organization Name (Optional)", placeholder="Your organization name")
    model_name = gr.Textbox(label="Model Name", placeholder="The name of your model on Hugging Face")
    make_private = gr.Checkbox(label="Make Repository Private", value=False)

    convert_button = gr.Button("Convert and Upload")
    output = gr.Markdown()

    convert_button.click(fn=main, inputs=[model_to_load, reference_model, output_path, hf_token, orgs_name, model_name, make_private], outputs=output)

demo.launch()
