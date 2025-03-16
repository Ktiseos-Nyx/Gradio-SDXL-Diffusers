import os
import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTextConfig
from safetensors.torch import load_file
from collections import OrderedDict
import requests
from urllib.parse import urlparse, unquote
from pathlib import Path
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from huggingface_hub import login, HfApi, hf_hub_download
from huggingface_hub.utils import validate_repo_id, HFValidationError
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import HfHubHTTPError
from accelerate import Accelerator
import re  # Import the 're' module


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
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies:\n{e.stderr}")
        raise

# ---------------------- UTILITY FUNCTIONS ----------------------

def download_model(model_path_or_url):
    """Downloads a model, handling URLs, HF repos, and local paths."""
    try:
        # 1. Check if it's a valid Hugging Face repo ID
        try:
            validate_repo_id(model_path_or_url)
            local_path = hf_hub_download(repo_id=model_path_or_url)
            return local_path
        except HFValidationError:
            pass

        # 2. Check if it's a URL
        if model_path_or_url.startswith("http://") or model_path_or_url.startswith("https://"):
            response = requests.get(model_path_or_url, stream=True)
            response.raise_for_status()

            parsed_url = urlparse(model_path_or_url)
            filename = os.path.basename(unquote(parsed_url.path))
            if not filename:
                filename = hashlib.sha256(model_path_or_url.encode()).hexdigest()

            cache_dir = os.path.join(HUGGINGFACE_HUB_CACHE, "downloads")
            os.makedirs(cache_dir, exist_ok=True)
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


def create_model_repo(api, user, orgs_name, model_name, make_private=False):
    """Creates a Hugging Face model repository, handling missing inputs and sanitizing the username."""

    print("---- create_model_repo Called ----")
    print(f"  user: {user}")
    print(f"  orgs_name: {orgs_name}")
    print(f"  model_name: {model_name}")

    if not model_name:
        model_name = f"converted-model-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        print(f"  Using default model_name: {model_name}")

    # --- Sanitize model_name and orgs_name ---
    if orgs_name:
        orgs_name = re.sub(r"[^a-zA-Z0-9._-]", "-", orgs_name)
        print(f"  Sanitized orgs_name: {orgs_name}")
    if model_name:
        model_name = re.sub(r"[^a-zA-Z0-9._-]", "-", model_name)
        print(f"  Sanitized model_name: {model_name}")


    if orgs_name:
        repo_id = f"{orgs_name}/{model_name.strip()}"
    elif user:
        sanitized_username = re.sub(r"[^a-zA-Z0-9._-]", "-", user['name'])
        print(f"  Original Username: {user['name']}")
        print(f"  Sanitized Username: {sanitized_username}")
        repo_id = f"{sanitized_username}/{model_name.strip()}"
    else:
        raise ValueError(
            "Must provide either an organization name or be logged in."
        )

    print(f"  repo_id: {repo_id}")

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=make_private)
        print(f"Model repo '{repo_id}' created.")
        return repo_id
    except Exception as e:
        print(f"Error creating repo: {e}")
        raise

def load_sdxl_checkpoint(checkpoint_path):
    """Loads checkpoint and extracts state dicts."""
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
        elif key.startswith("condition_model.model.text_encoder."):  # First Text Encoder
            text_encoder1_state[key.replace("condition_model.model.text_encoder.", "")] = value.to(torch.float16)
        elif key.startswith("condition_model.model.text_encoder_2."):  # Second Text Encoder
            text_encoder2_state[key.replace("condition_model.model.text_encoder_2.", "")] = value.to(torch.float16)
        elif key.startswith("model.diffusion_model."):  # UNet
            unet_state[key.replace("model.diffusion_model.", "")] = value.to(torch.float16)

    return text_encoder1_state, text_encoder2_state, vae_state, unet_state



def build_diffusers_model(text_encoder1_state, text_encoder2_state, vae_state, unet_state, reference_model_path=None):
    """Builds Diffusers components using accelerate for low-memory loading."""
    if not reference_model_path:
        reference_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    # Initialize the Accelerator
    accelerator = Accelerator(mixed_precision="fp16") # Use mixed precision
    device = accelerator.device

    # Load configurations from the reference model
    config_text_encoder1 = CLIPTextConfig.from_pretrained(
        reference_model_path, subfolder="text_encoder"
    )
    config_text_encoder2 = CLIPTextConfig.from_pretrained(
       reference_model_path, subfolder="text_encoder_2"
    )

    # Use from_pretrained with device_map and low_cpu_mem_usage for all components
    text_encoder1 = CLIPTextModel.from_pretrained(reference_model_path, subfolder="text_encoder", config=config_text_encoder1, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(device)
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(reference_model_path, subfolder="text_encoder_2", config=config_text_encoder2, low_cpu_mem_usage=True,  torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(reference_model_path, subfolder="vae", low_cpu_mem_usage=True,  torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel.from_pretrained(reference_model_path, subfolder="unet", low_cpu_mem_usage=True,  torch_dtype=torch.float16).to(device)


    # Load state dicts with strict=False
    text_encoder1.load_state_dict(text_encoder1_state, strict=False)
    text_encoder2.load_state_dict(text_encoder2_state, strict=False)
    vae.load_state_dict(vae_state, strict=False)
    unet.load_state_dict(unet_state, strict=False)

    return text_encoder1, text_encoder2, vae, unet

def convert_and_save_sdxl_to_diffusers(checkpoint_path_or_url, output_path, reference_model_path):
    """Converts and saves the checkpoint to Diffusers format."""
    checkpoint_path = download_model(checkpoint_path_or_url)
    text_encoder1_state, text_encoder2_state, vae_state, unet_state = load_sdxl_checkpoint(checkpoint_path)
    text_encoder1, text_encoder2, vae, unet = build_diffusers_model(
        text_encoder1_state, text_encoder2_state, vae_state, unet_state, reference_model_path
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

    pipeline.save_pretrained(output_path)
    print(f"Model saved as Diffusers format: {output_path}")

# ---------------------- MAIN FUNCTION (with Debugging Prints) ----------------------

def main(
    model_to_load,
    reference_model,
    output_path,
    hf_token,
    orgs_name,
    model_name,
    make_private,
):
    """Main function: SDXL checkpoint to Diffusers, always fp16."""

    print("---- Main Function Called ----")
    print(f"  model_to_load: {model_to_load}")
    print(f"  reference_model: {reference_model}")
    print(f"  output_path: {output_path}")
    print(f"  hf_token: {hf_token}")
    print(f"  orgs_name: {orgs_name}")
    print(f"  model_name: {model_name}")
    print(f"  make_private: {make_private}")

    # --- Force Login at the Beginning of main() ---
    try:
        login(token=hf_token, add_to_git_credential=True)
        api = HfApi()
        user = api.whoami()  # Get logged-in user info
        print(f"  Logged-in user: {user}")
    except Exception as e:
        error_message = f"Error during login: {e} Ensure a valid WRITE token is provided."
        print(f"---- Main Function Error: {error_message} ----")
        return error_message

    # --- Strip Whitespace and Sanitize from Inputs ---
    model_to_load = model_to_load.strip()
    reference_model = reference_model.strip()
    output_path = output_path.strip()
    hf_token = hf_token.strip()  # Even though it's a password field
    orgs_name = orgs_name.strip() if orgs_name else ""
    model_name = model_name.strip() if model_name else ""

    # --- Sanitize model_name and orgs_name ---
    if orgs_name:
        orgs_name = re.sub(r"[^a-zA-Z0-9._-]", "-", orgs_name)
    if model_name:
        model_name = re.sub(r"[^a-zA-Z0-9._-]", "-", model_name)

    try:
        convert_and_save_sdxl_to_diffusers(model_to_load, output_path, reference_model)

        # --- Create Repo and Upload (Simplified) ---
        if not model_name:
            model_name = f"converted-model-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            print(f"Using default model_name: {model_name}")

        if orgs_name:
            repo_id = f"{orgs_name}/{model_name}"
        elif user:
            # Sanitize username here as well:
            sanitized_username = re.sub(r"[^a-zA-Z0-9._-]", "-", user['name'])
            print(f"  Sanitized Username: {sanitized_username}")
            repo_id = f"{sanitized_username}/{model_name}"

        else:  # Should never happen because of login, but good practice
            raise ValueError("Must provide either an organization name or be logged in.")
        print(f"repo_id = {repo_id}")
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=make_private)
            print(f"Model repo '{repo_id}' created.")
        except Exception as e:
            print(f"Error in creating model repo: {e}")
            raise

        api.upload_folder(folder_path=output_path, repo_id=repo_id)
        print(f"Model uploaded to: https://huggingface.co/{repo_id}")

        result = "Conversion and upload completed successfully!"
        print(f"---- Main Function Successful: {result} ----")
        return result

    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(f"---- Main Function Error: {error_message} ----")
        return error_message

# ---------------------- GRADIO INTERFACE ----------------------

css = """
#main-container {
    display: flex;
    flex-direction: column;
    font-family: 'Arial', sans-serif;
    font-size: 16px;
    color: #333;
}
#convert-button {
    margin-top: 1em;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
    # 🎨 SDXL Model Converter
    Convert SDXL checkpoints to Diffusers format (FP16, CPU-only).

    ### 📥 Input Sources Supported:
    - Local model files (.safetensors, .ckpt)
    - Direct URLs to model files
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

    with gr.Row():
        with gr.Column():
            model_to_load = gr.Textbox(
                label="SDXL Checkpoint (Path, URL, or HF Repo)",
                placeholder="Path, URL, or Hugging Face Repo ID (e.g., my-org/my-model or my-org/my-model/file.safetensors)",
            )
            reference_model = gr.Textbox(
                label="Reference Diffusers Model (Optional)",
                placeholder="e.g., stabilityai/stable-diffusion-xl-base-1.0 (Leave blank for default)",
            )
            output_path = gr.Textbox(label="Output Path (Diffusers Format)", value="output")
            hf_token = gr.Textbox(label="Hugging Face Token", placeholder="Your Hugging Face write token", type="password")
            orgs_name = gr.Textbox(label="Organization Name (Optional)", placeholder="Your organization name")
            model_name = gr.Textbox(label="Model Name", placeholder="The name of your model on Hugging Face")
            make_private = gr.Checkbox(label="Make Repository Private", value=False)
            convert_button = gr.Button("Convert and Upload")

        with gr.Column(variant="panel"):
            output = gr.Markdown(container=True)

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
