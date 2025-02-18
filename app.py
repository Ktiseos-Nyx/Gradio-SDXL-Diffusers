# Core functionality
import os
import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextConfig

# Model handling
from safetensors.torch import load_file
from collections import OrderedDict

# Utilities
import re
import json
import gdown
import requests
import subprocess
from urllib.parse import urlparse, unquote
from pathlib import Path
import tempfile
from tqdm import tqdm
import psutil
import math
import shutil
import hashlib
from datetime import datetime
from typing import Dict, List, Optional

# Hugging Face integration
from huggingface_hub import login, HfApi
from types import SimpleNamespace

# Remove unused imports
# import os
# import gradio as gr
# import torch
# from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
# from transformers import CLIPTextModel, CLIPTextConfig
# from safetensors.torch import load_file
# from collections import OrderedDict
# import re
# import json
# import gdown
# import requests
# import subprocess
# from urllib.parse import urlparse, unquote
# from pathlib import Path
# import tempfile
# from tqdm import tqdm
# import psutil
# import math
# import shutil
# import hashlib
# from datetime import datetime
# from typing import Dict, List, Optional
# from huggingface_hub import login, HfApi
# from types import SimpleNamespace

# ---------------------- UTILITY FUNCTIONS ----------------------

def is_valid_url(url):
    """Checks if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception as e:
        print(f"Error checking URL validity: {e}")
        return False

def get_filename(url):
    """Extracts the filename from a URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        if 'content-disposition' in response.headers:
            content_disposition = response.headers['content-disposition']
            filename = re.findall('filename="?([^";]+)"?', content_disposition)[0]
        else:
            url_path = urlparse(url).path
            filename = unquote(os.path.basename(url_path))

        return filename
    except Exception as e:
        print(f"Error getting filename from URL: {e}")
        return None

def get_supported_extensions():
    """Returns a tuple of supported model file extensions."""
    return tuple([".ckpt", ".safetensors", ".pt", ".pth"])

def download_model(url, dst, output_widget):
    """Downloads a model from a URL to the specified destination."""
    filename = get_filename(url)
    filepath = os.path.join(dst, filename)
    try:
        if "drive.google.com" in url:
            gdown = gdown_download(url, dst, filepath)
        else:
            if "huggingface.co" in url:
                if "/blob/" in url:
                    url = url.replace("/blob/", "/resolve/")
            subprocess.run(["aria2c","-x 16",url,"-d",dst,"-o",filename])
        return filepath
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def determine_load_checkpoint(model_to_load):
    """Determines if the model to load is a checkpoint, Diffusers model, or URL."""
    try:
        if is_valid_url(model_to_load) and (model_to_load.endswith(get_supported_extensions())):
            return True
        elif model_to_load.endswith(get_supported_extensions()):
            return True
        elif os.path.isdir(model_to_load):
            required_folders = {"unet", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler", "vae"}
            if required_folders.issubset(set(os.listdir(model_to_load))) and os.path.isfile(os.path.join(model_to_load, "model_index.json")):
                return False
    except Exception as e:
        print(f"Error determining load checkpoint: {e}")
    return None  # handle this case as required

def create_model_repo(api, user, orgs_name, model_name, make_private=False):
    """Creates a Hugging Face model repository if it doesn't exist."""
    try:
        if orgs_name == "":
            repo_id = user["name"] + "/" + model_name.strip()
        else:
            repo_id = orgs_name + "/" + model_name.strip()

        validate_repo_id(repo_id)
        api.create_repo(repo_id=repo_id, repo_type="model", private=make_private)
        print(f"Model repo '{repo_id}' didn't exist, creating repo")
    except HfHubHTTPError as e:
        print(f"Model repo '{repo_id}' exists, skipping create repo")

    print(f"Model repo '{repo_id}' link: https://huggingface.co/{repo_id}\n")

    return repo_id

def is_diffusers_model(model_path):
    """Checks if a given path is a valid Diffusers model directory."""
    try:
        required_folders = {"unet", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler", "vae"}
        return required_folders.issubset(set(os.listdir(model_path))) and os.path.isfile(os.path.join(model_path, "model_index.json"))
    except Exception as e:
        print(f"Error checking if model is a Diffusers model: {e}")
        return False

# ---------------------- MODEL UTIL (From library.sdxl_model_util) ----------------------

def load_models_from_sdxl_checkpoint(sdxl_base_id, checkpoint_path, device):
    """Loads SDXL model components from a checkpoint file."""
    try:
        text_encoder1 = CLIPTextModel.from_pretrained(sdxl_base_id, subfolder="text_encoder").to(device)
        text_encoder2 = CLIPTextModel.from_pretrained(sdxl_base_id, subfolder="text_encoder_2").to(device)
        vae = AutoencoderKL.from_pretrained(sdxl_base_id, subfolder="vae").to(device)
        unet = UNet2DConditionModel.from_pretrained(sdxl_base_id, subfolder="unet").to(device)
        unet = unet

        ckpt_state_dict = torch.load(checkpoint_path, map_location=device)

        o = OrderedDict()
        for key in list(ckpt_state_dict.keys()):
            o[key.replace("module.", "")] = ckpt_state_dict[key]
        del ckpt_state_dict

        print("Applying weights to text encoder 1:")
        text_encoder1.load_state_dict({
            '.'.join(key.split('.')[1:]): o[key] for key in list(o.keys()) if key.startswith("first_stage_model.cond_stage_model.model.transformer")
        }, strict=False)
        print("Applying weights to text encoder 2:")
        text_encoder2.load_state_dict({
            '.'.join(key.split('.')[1:]): o[key] for key in list(o.keys()) if key.startswith("cond_stage_model.model.transformer")
        }, strict=False)
        print("Applying weights to VAE:")
        vae.load_state_dict({
            '.'.join(key.split('.')[2:]): o[key] for key in list(o.keys()) if key.startswith("first_stage_model.model")
        }, strict=False)
        print("Applying weights to UNet:")
        unet.load_state_dict({
            key: o[key] for key in list(o.keys()) if key.startswith("model.diffusion_model")
        }, strict=False)

        logit_scale = None #Not used here!
        global_step = None #Not used here!
        return text_encoder1, text_encoder2, vae, unet, logit_scale, global_step
    except Exception as e:
        print(f"Error loading models from checkpoint: {e}")
        return None

def save_stable_diffusion_checkpoint(save_path, text_encoder1, text_encoder2, unet, epoch, global_step, ckpt_info, vae, logit_scale, save_dtype):
    """Saves the stable diffusion checkpoint."""
    weights = OrderedDict()
    text_encoder1_dict = text_encoder1.state_dict()
    text_encoder2_dict = text_encoder2.state_dict()
    unet_dict = unet.state_dict()
    vae_dict = vae.state_dict()

    def replace_key(key):
        key = "cond_stage_model.model.transformer." + key
        return key

    print("Merging text encoder 1")
    for key in tqdm(list(text_encoder1_dict.keys())):
        weights["first_stage_model.cond_stage_model.model.transformer." + key] = text_encoder1_dict[key].to(save_dtype)

    print("Merging text encoder 2")
    for key in tqdm(list(text_encoder2_dict.keys())):
        weights[replace_key(key)] = text_encoder2_dict[key].to(save_dtype)

    print("Merging vae")
    for key in tqdm(list(vae_dict.keys())):
        weights["first_stage_model.model." + key] = vae_dict[key].to(save_dtype)

    print("Merging unet")
    for key in tqdm(list(unet_dict.keys())):
        weights["model.diffusion_model." + key] = unet_dict[key].to(save_dtype)

    info = {"epoch": epoch, "global_step": global_step}
    if ckpt_info is not None:
        info.update(ckpt_info)

    if logit_scale is not None:
        info["logit_scale"] = logit_scale.item()

    torch.save({"state_dict": weights, "info": info}, save_path)

    key_count = len(weights.keys())
    del weights
    del text_encoder1_dict, text_encoder2_dict, unet_dict, vae_dict
    return key_count

def save_diffusers_checkpoint(save_path, text_encoder1, text_encoder2, unet, reference_model, vae, trim_if_model_exists, save_dtype):
    """Saves the SDXL model as a Diffusers model."""
    print("Saving SDXL as Diffusers format to:", save_path)
    print("SDXL Text Encoder 1 to:", os.path.join(save_path, "text_encoder"))
    text_encoder1.save_pretrained(os.path.join(save_path, "text_encoder"))

    print("SDXL Text Encoder 2 to:", os.path.join(save_path, "text_encoder_2"))
    text_encoder2.save_pretrained(os.path.join(save_path, "text_encoder_2"))

    print("SDXL VAE to:", os.path.join(save_path, "vae"))
    vae.save_pretrained(os.path.join(save_path, "vae"))

    print("SDXL UNet to:", os.path.join(save_path, "unet"))
    unet.save_pretrained(os.path.join(save_path, "unet"))

    if reference_model is not None:
        print(f"Copying scheduler from {reference_model}")
        scheduler_src = StableDiffusionXLPipeline.from_pretrained(reference_model, torch_dtype=torch.float16).scheduler
        torch.save(scheduler_src.config, os.path.join(save_path, "scheduler", "scheduler_config.json"))
    else:
        print(f"No reference Model. Copying scheduler from original model.")
        scheduler_src = StableDiffusionXLPipeline.from_pretrained(reference_model, torch_dtype=torch.float16).scheduler
        scheduler_src.save_pretrained(save_path)

    if trim_if_model_exists:
        print("Trim Complete")

# ---------------------- CONVERSION AND UPLOAD FUNCTIONS ----------------------

def load_sdxl_model(args, is_load_checkpoint, load_dtype, output_widget):
    """Loads the SDXL model from a checkpoint or Diffusers model."""
    model_load_message = "checkpoint" if is_load_checkpoint else "Diffusers" + (" as fp16" if args.fp16 else "")
    with output_widget:
        print(f"Loading {model_load_message}: {args.model_to_load}")

    if is_load_checkpoint:
        loaded_model_data = load_from_sdxl_checkpoint(args, output_widget)
    else:
        loaded_model_data = load_sdxl_from_diffusers(args, load_dtype)

    return loaded_model_data

def load_from_sdxl_checkpoint(args, output_widget):
    """Loads the SDXL model components from a checkpoint file (placeholder)."""
    text_encoder1, text_encoder2, vae, unet = None, None, None, None
    device = "cpu"
    if is_valid_url(args.model_to_load):
        tmp_model_name = "download"
        download_dst_dir = tempfile.mkdtemp()
        model_path = download_model(args.model_to_load, download_dst_dir, output_widget)
        #model_path = os.path.join(download_dst_dir,tmp_model_name)
        if model_path == None:
            with output_widget:
                print("Loading from Checkpoint failed, the request could not be completed")
            return text_encoder1, text_encoder2, vae, unet
        else:
            # Implement Load model from ckpt or safetensors
            try:
                text_encoder1, text_encoder2, vae, unet, _, _ = load_models_from_sdxl_checkpoint(
                    "sdxl_base_v1-0", model_path, device
                )
                return text_encoder1, text_encoder2, vae, unet
            except Exception as e:
                print(f"Could not load SDXL from checkpoint due to: \n{e}")
                return text_encoder1, text_encoder2, vae, unet

            with output_widget:
                print(f"Loading from Checkpoint from URL needs to be implemented - using {model_path}")
    else:
        # Implement Load model from ckpt or safetensors
        try:
            text_encoder1, text_encoder2, vae, unet, _, _ = load_models_from_sdxl_checkpoint(
                "sdxl_base_v1-0", args.model_to_load, device
            )
            return text_encoder1, text_encoder2, vae, unet
        except Exception as e:
            print(f"Could not load SDXL from checkpoint due to: \n{e}")
            return text_encoder1, text_encoder2, vae, unet

        with output_widget:
            print("Loading from Checkpoint needs to be implemented.")

    return text_encoder1, text_encoder2, vae, unet

def load_sdxl_from_diffusers(args, load_dtype):
    """Loads an SDXL model from a Diffusers model directory."""
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model_to_load, torch_dtype=load_dtype, tokenizer=None, tokenizer_2=None, scheduler=None
    )
    text_encoder1 = pipeline.text_encoder
    text_encoder2 = pipeline.text_encoder_2
    vae = pipeline.vae
    unet = pipeline.unet

    return text_encoder1, text_encoder2, vae, unet

def convert_and_save_sdxl_model(args, is_save_checkpoint, loaded_model_data, save_dtype, output_widget):
    """Converts and saves the SDXL model as either a checkpoint or a Diffusers model."""
    text_encoder1, text_encoder2, vae, unet = loaded_model_data
    model_save_message = "checkpoint" + ("" if save_dtype is None else f" in {save_dtype}") if is_save_checkpoint else "Diffusers"

    with output_widget:
        print(f"Converting and saving as {model_save_message}: {args.model_to_save}")

    if is_save_checkpoint:
        save_sdxl_as_checkpoint(args, text_encoder1, text_encoder2, vae, unet, save_dtype, output_widget)
    else:
        save_sdxl_as_diffusers(args, text_encoder1, text_encoder2, vae, unet, save_dtype, output_widget)

def save_sdxl_as_checkpoint(args, text_encoder1, text_encoder2, vae, unet, save_dtype, output_widget):
    """Saves the SDXL model components as a checkpoint file (placeholder)."""
    logit_scale = None
    ckpt_info = None

    key_count = save_stable_diffusion_checkpoint(
        args.model_to_save, text_encoder1, text_encoder2, unet, args.epoch, args.global_step, ckpt_info, vae, logit_scale, save_dtype
        )
    with output_widget:
        print(f"Model saved. Total converted state_dict keys: {key_count}")

def save_sdxl_as_diffusers(args, text_encoder1, text_encoder2, vae, unet, save_dtype, output_widget):
    """Saves the SDXL model as a Diffusers model."""
    with output_widget:
        reference_model_message = args.reference_model if args.reference_model is not None else 'default model'
        print(f"Copying scheduler/tokenizer config from: {reference_model_message}")

    # Save diffusers pipeline
    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder1,
        text_encoder_2=text_encoder2,
        unet=unet,
        scheduler=None,  # Replace None if there is a scheduler
        tokenizer=None,  # Replace None if there is a tokenizer
        tokenizer_2=None  # Replace None if there is a tokenizer_2
    )

    pipeline.save_pretrained(args.model_to_save)

    with output_widget:
        print(f"Model saved as {save_dtype}.")

def get_save_dtype(precision):
    """
    Convert precision string to torch dtype
    """
    if precision == "float32" or precision == "fp32":
        return torch.float32
    elif precision == "float16" or precision == "fp16":
        return torch.float16
    elif precision == "bfloat16" or precision == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision: {precision}")

def get_file_size(file_path):
    """Get file size in GB."""
    try:
        size_bytes = Path(file_path).stat().st_size
        return size_bytes / (1024 * 1024 * 1024)  # Convert to GB
    except:
        return None

def get_available_memory():
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / (1024 * 1024 * 1024)

def estimate_memory_requirements(model_path, precision):
    """Estimate memory requirements for model conversion."""
    try:
        # Base memory requirement for SDXL
        base_memory = 8  # GB
        
        # Get model size if local file
        model_size = get_file_size(model_path) if not is_valid_url(model_path) else None
        
        # Adjust for precision
        memory_multiplier = 1.0 if precision in ["float16", "fp16", "bfloat16", "bf16"] else 2.0
        
        # Calculate total required memory
        required_memory = (base_memory + (model_size if model_size else 12)) * memory_multiplier
        
        return required_memory
    except:
        return 16  # Default safe estimate

def validate_model(model_path, precision):
    """
    Validate the model before conversion.
    Returns (is_valid, message)
    """
    try:
        # Check if it's a URL
        if is_valid_url(model_path):
            try:
                response = requests.head(model_path)
                if response.status_code != 200:
                    return False, "❌ Invalid URL or model not accessible"
                if 'content-length' in response.headers:
                    size_gb = int(response.headers['content-length']) / (1024 * 1024 * 1024)
                    if size_gb < 0.1 and not model_path.endswith(('.ckpt', '.safetensors')):
                        return False, "❌ File too small to be a valid model"
            except:
                return False, "❌ Error checking URL"
        
        # Check if it's a local file
        elif not model_path.startswith("stabilityai/") and not Path(model_path).exists():
            return False, "❌ Model file not found"
        
        # Check available memory
        available_memory = get_available_memory()
        required_memory = estimate_memory_requirements(model_path, precision)
        
        if available_memory < required_memory:
            return True, f"⚠️ Insufficient memory detected. Need {math.ceil(required_memory)}GB, but only {math.ceil(available_memory)}GB available"
        
        # Memory warning
        memory_message = ""
        if available_memory < required_memory * 1.5:
            memory_message = "⚠️ Memory is tight. Consider closing other applications."
        
        return True, f"✅ Model validated successfully. {memory_message}"
    
    except Exception as e:
        return False, f"❌ Validation error: {str(e)}"

def cleanup_temp_files(directory=None):
    """Clean up temporary files after conversion."""
    try:
        if directory:
            shutil.rmtree(directory, ignore_errors=True)
        # Clean up other temp files
        temp_pattern = "*.tmp"
        for temp_file in Path(".").glob(temp_pattern):
            temp_file.unlink()
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

def convert_model(model_to_load, save_precision_as, epoch, global_step, reference_model, fp16, use_xformers, hf_token, orgs_name, model_name, make_private, output_widget):
    """Convert the model between different formats."""
    temp_dir = None
    history = ConversionHistory()
    
    try:
        print("Starting model conversion...")
        update_progress(output_widget, "⏳ Initializing conversion process...", 0)
        
        # Get optimization suggestions
        available_memory = get_available_memory()
        auto_suggestions = get_auto_optimization_suggestions(model_to_load, save_precision_as, available_memory)
        history_suggestions = history.get_optimization_suggestions(model_to_load)
        
        # Display suggestions
        if auto_suggestions or history_suggestions:
            print("\n🔍 Optimization Suggestions:")
            for suggestion in auto_suggestions + history_suggestions:
                print(suggestion)
            print("\n")
        
        # Validate model
        is_valid, message = validate_model(model_to_load, save_precision_as)
        if not is_valid:
            raise ValueError(message)
        print(message)
        
        args = SimpleNamespace()
        args.model_to_load = model_to_load
        args.save_precision_as = save_precision_as
        args.epoch = epoch
        args.global_step = global_step
        args.reference_model = reference_model
        args.fp16 = fp16
        args.use_xformers = use_xformers

        update_progress(output_widget, "🔍 Validating input model...", 10)
        args.model_to_save = increment_filename(os.path.splitext(args.model_to_load)[0] + ".safetensors")
        
        save_dtype = get_save_dtype(save_precision_as)
        
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="sdxl_conversion_")
        
        update_progress(output_widget, "📥 Loading model components...", 30)
        is_load_checkpoint = determine_load_checkpoint(args.model_to_load)
        if is_load_checkpoint is None:
            raise ValueError("Invalid model format or path")
            
        update_progress(output_widget, "🔄 Converting model...", 50)
        loaded_model_data = load_sdxl_model(args, is_load_checkpoint, save_dtype, output_widget)
        
        update_progress(output_widget, "💾 Saving converted model...", 80)
        is_save_checkpoint = args.model_to_save.endswith(get_supported_extensions())
        result = convert_and_save_sdxl_model(args, is_save_checkpoint, loaded_model_data, save_dtype, output_widget)
        
        update_progress(output_widget, "✅ Conversion completed!", 100)
        print(f"Model conversion completed. Saved to: {args.model_to_save}")
        
        # Verify the converted model
        is_valid, verify_message = verify_model_structure(args.model_to_save)
        if not is_valid:
            raise ValueError(verify_message)
        print(verify_message)
        
        # Record successful conversion
        history.add_entry(
            model_to_load,
            {
                'precision': save_precision_as,
                'fp16': fp16,
                'epoch': epoch,
                'global_step': global_step
            },
            True,
            "Conversion completed successfully"
        )
        
        cleanup_temp_files(temp_dir)
        return result
        
    except Exception as e:
        if temp_dir:
            cleanup_temp_files(temp_dir)
        
        # Record failed conversion
        history.add_entry(
            model_to_load,
            {
                'precision': save_precision_as,
                'fp16': fp16,
                'epoch': epoch,
                'global_step': global_step
            },
            False,
            str(e)
        )
        
        error_msg = f"❌ Error during model conversion: {str(e)}"
        print(error_msg)
        return error_msg

def update_progress(output_widget, message, progress):
    """Update the progress bar and message in the UI."""
    progress_bar = "▓" * (progress // 5) + "░" * ((100 - progress) // 5)
    print(f"{message}\n[{progress_bar}] {progress}%")

class ConversionHistory:
    def __init__(self, history_file="conversion_history.json"):
        self.history_file = history_file
        self.history = self._load_history()

    def _load_history(self) -> List[Dict]:
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def add_entry(self, model_path: str, settings: Dict, success: bool, message: str):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'settings': settings,
            'success': success,
            'message': message
        }
        self.history.append(entry)
        self._save_history()

    def get_optimization_suggestions(self, model_path: str) -> List[str]:
        """Analyze history and provide optimization suggestions."""
        suggestions = []
        similar_conversions = [h for h in self.history if h['model_path'] == model_path]
        
        if similar_conversions:
            success_rate = sum(1 for h in similar_conversions if h['success']) / len(similar_conversions)
            if success_rate < 1.0:
                failed_attempts = [h for h in similar_conversions if not h['success']]
                if any('memory' in h['message'].lower() for h in failed_attempts):
                    suggestions.append("⚠️ Previous attempts had memory issues. Consider using fp16 precision.")
                if any('timeout' in h['message'].lower() for h in failed_attempts):
                    suggestions.append("⚠️ Previous attempts timed out. Try breaking down the conversion process.")

        return suggestions

def verify_model_structure(model_path: str) -> tuple[bool, str]:
    """Verify the structure of the converted model."""
    try:
        if model_path.endswith('.safetensors'):
            # Verify safetensors structure
            with safe_open(model_path, framework="pt") as f:
                if not f.keys():
                    return False, "❌ Invalid safetensors file: no tensors found"
        
        # Check for essential components
        required_keys = ["model.diffusion_model", "first_stage_model"]
        missing_keys = []
        
        # Load and check key components
        state_dict = load_file(model_path)
        for key in required_keys:
            if not any(k.startswith(key) for k in state_dict.keys()):
                missing_keys.append(key)
        
        if missing_keys:
            return False, f"❌ Missing essential components: {', '.join(missing_keys)}"
        
        return True, "✅ Model structure verified successfully"
    except Exception as e:
        return False, f"❌ Model verification failed: {str(e)}"

def get_auto_optimization_suggestions(model_path: str, precision: str, available_memory: float) -> List[str]:
    """Generate automatic optimization suggestions based on model and system characteristics."""
    suggestions = []
    
    # Memory-based suggestions
    if available_memory < 16:
        suggestions.append("💡 Limited memory detected. Consider these options:")
        suggestions.append("   - Use fp16 precision to reduce memory usage")
        suggestions.append("   - Close other applications before conversion")
        suggestions.append("   - Use a machine with more RAM if available")
    
    # Precision-based suggestions
    if precision == "float32" and available_memory < 32:
        suggestions.append("💡 Consider using fp16 precision for better memory efficiency")
    
    # Model size-based suggestions
    model_size = get_file_size(model_path) if not is_valid_url(model_path) else None
    if model_size and model_size > 10:
        suggestions.append("💡 Large model detected. Recommendations:")
        suggestions.append("   - Ensure stable internet connection for URL downloads")
        suggestions.append("   - Consider breaking down the conversion process")
    
    return suggestions

def upload_to_huggingface(model_path, hf_token, orgs_name, model_name, make_private):
    """Uploads a model to the Hugging Face Hub."""
    try:
        # Login to Hugging Face
        login(hf_token, add_to_git_credential=True)
        
        # Prepare model upload
        if not os.path.exists(model_path):
            raise ValueError("Model path does not exist.")
        
        # Check if repo already exists
        api = HfApi()
        repo_id = f"{orgs_name}/{model_name}" if orgs_name else model_name
        try:
            api.repo_info(repo_id)
            print(f"⚠️ Repository '{repo_id}' already exists. Proceeding with upload.")
        except Exception:
            if make_private:
                api.create_repo(repo_id, private=True)
            else:
                api.create_repo(repo_id)
        
        # Push model files
        api.upload_folder(
            folder_path=model_path,
            path_in_repo="",
            repo_id=repo_id,
            commit_message=f"Upload model: {model_name}",
            ignore_patterns=".ipynb_checkpoints",
        )
        
        print(f"Model uploaded to: https://huggingface.co/{repo_id}")
        return f"Model uploaded to: https://huggingface.co/{repo_id}"
    except Exception as e:
        error_msg = f"❌ Error during upload: {str(e)}"
        print(error_msg)
        return error_msg

# ---------------------- GRADIO INTERFACE ----------------------

def main(model_to_load, save_precision_as, epoch, global_step, reference_model, fp16, use_xformers, hf_token, orgs_name, model_name, make_private):
  """Main function orchestrating the entire process."""
  output = gr.Markdown()

  # Create tempdir, will only be there for the function
  with tempfile.TemporaryDirectory() as output_path:
    conversion_output = convert_model(model_to_load, save_precision_as, epoch, global_step, reference_model, fp16, use_xformers, hf_token, orgs_name, model_name, make_private, output)

    upload_output = upload_to_huggingface(output_path, hf_token, orgs_name, model_name, make_private)

    # Return a combined output
    return f"{conversion_output}\n\n{upload_output}"

def increment_filename(filename):
    """
    If a file exists, add a number to the filename to make it unique.
    Example: if test.txt exists, return test(1).txt
    """
    if not os.path.exists(filename):
        return filename

    directory = os.path.dirname(filename)
    name, ext = os.path.splitext(os.path.basename(filename))
    counter = 1

    while True:
        new_name = os.path.join(directory, f"{name}({counter}){ext}")
        if not os.path.exists(new_name):
            return new_name
        counter += 1

with gr.Blocks(css="#main-container { display: flex; flex-direction: column; height: 100vh; justify-content: space-between; font-family: 'Arial', sans-serif; font-size: 16px; color: #333; } #convert-button { margin-top: auto; }") as demo:
    gr.Markdown("""
    # 🎨 SDXL Model Converter
    Convert SDXL models between different formats and precisions. Works on CPU!
    
    ### 📥 Input Sources Supported:
    - Local model files (.safetensors, .ckpt, etc.)
    - Direct URLs to model files
    - Hugging Face model repositories (e.g., 'stabilityai/stable-diffusion-xl-base-1.0')
    
    ### ℹ️ Important Notes:
    - This tool runs on CPU, though conversion might be slower than on GPU
    - For Hugging Face uploads, you need a **WRITE** token (not a read token)
    - Get your HF token here: https://huggingface.co/settings/tokens
    
    ### 💾 Memory Usage Tips:
    - Use FP16 precision when possible to reduce memory usage
    - Close other applications during conversion
    - For large models, ensure you have at least 16GB of RAM
    """)
    with gr.Row():
        with gr.Column():
            model_to_load = gr.Textbox(
                label="Model Path/URL/HF Repo",
                placeholder="Enter local path, URL, or Hugging Face model ID (e.g., stabilityai/stable-diffusion-xl-base-1.0)",
                type="text"
            )
            
            save_precision_as = gr.Dropdown(
                choices=["float32", "float16", "bfloat16"],
                value="float16",
                label="Save Precision",
                info="Choose model precision (float16 recommended for most cases)"
            )

            with gr.Row():
                epoch = gr.Number(
                    value=0,
                    label="Epoch",
                    precision=0,
                    info="Optional: Set epoch number for the saved model"
                )
                global_step = gr.Number(
                    value=0,
                    label="Global Step",
                    precision=0,
                    info="Optional: Set training step for the saved model"
                )

            reference_model = gr.Textbox(
                label="Reference Model (Optional)",
                placeholder="Path to reference model for scheduler config",
                info="Optional: Used to copy scheduler configuration"
            )

            fp16 = gr.Checkbox(
                label="Load in FP16",
                value=True,
                info="Load model in half precision (recommended for CPU usage)"
            )

            use_xformers = gr.Checkbox(
                label="Enable Memory-Efficient Attention",
                value=False,
                info="Enable xFormers for reduced memory usage during conversion"
            )

            # Hugging Face Upload Section
            gr.Markdown("### Upload to Hugging Face (Optional)")
            
            hf_token = gr.Textbox(
                label="Hugging Face Token",
                placeholder="Enter your WRITE token from huggingface.co/settings/tokens",
                type="password",
                info=" Must be a WRITE token, not a read token!"
            )

            with gr.Row():
                orgs_name = gr.Textbox(
                    label="Organization Name",
                    placeholder="Optional: Your organization name",
                    info="Leave empty to use your personal account"
                )
                model_name = gr.Textbox(
                    label="Model Name",
                    placeholder="Name for your uploaded model",
                    info="The name your model will have on Hugging Face"
                )

            make_private = gr.Checkbox(
                label="Make Private",
                value=True,
                info="Keep the uploaded model private on Hugging Face"
            )

        with gr.Column():
            output = gr.Markdown(label="Output")
            convert_btn = gr.Button("Convert Model", variant="primary", elem_id="convert-button")
            convert_btn.click(
                fn=main,
                inputs=[
                    model_to_load,
                    save_precision_as,
                    epoch,
                    global_step,
                    reference_model,
                    fp16,
                    use_xformers,
                    hf_token,
                    orgs_name,
                    model_name,
                    make_private
                ],
                outputs=output
            )

demo.launch()