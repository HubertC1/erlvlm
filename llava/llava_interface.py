import torch
import subprocess
import json
import base64
import io
from PIL import Image
import os
import random

# Global variables to hold the model status
llava_model = None
device = None

def initialize_llava_model(cache_dir: str):
    """
    Initializes the LLaVA model using Ollama as a fallback approach.
    This provides a working LLaVA interface without dependency issues.

    Args:
        cache_dir (str): The directory to cache the downloaded model weights.
    """
    global llava_model, device

    if llava_model is not None:
        # Model is already initialized
        return

    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    print(f"INFO: Using cache directory: {cache_dir}")

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Using device: {device}")

    # Check if Ollama is available
    try:
        result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
        if result.returncode == 0:
            print("INFO: Ollama found, checking for LLaVA models...")
            
            # Try to use Ollama with LLaVA
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if 'llava' in result.stdout.lower() or 'llama-vision' in result.stdout.lower():
                llava_model = "ollama"
                print("INFO: LLaVA model available via Ollama")
                return
            else:
                print("INFO: No LLaVA models found in Ollama, will download...")
                # Try to pull a LLaVA model
                result = subprocess.run(['ollama', 'pull', 'llava'], capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    llava_model = "ollama"
                    print("INFO: LLaVA model downloaded via Ollama")
                    return
                else:
                    print(f"WARNING: Failed to download LLaVA via Ollama: {result.stderr}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"INFO: Ollama not available: {e}")

def generate_text_from_image(image, prompt: str) -> str:
    """
    Generates text from an image and a prompt using the LLaVA model.

    Args:
        image: The input image (PIL.Image.Image or None for text-only queries).
        prompt (str): The text prompt.

    Returns:
        str: The generated text from the model.
    """
    global llava_model, device

    if llava_model is None:
        raise RuntimeError("LLaVA model not initialized. Please call initialize_llava_model() first.")

    print(f"[LLaVA] Processing query: {prompt[:80]}..." if len(prompt) > 80 else f"[LLaVA] Processing query: {prompt}")
    
    if llava_model == "ollama":
        print("generating via ollama for real")
        return _generate_with_ollama(image, prompt)
    else:
        raise RuntimeError(f"Unknown LLaVA model type: {llava_model}")

def _generate_with_ollama(image, prompt: str) -> str:
    """Generate response using Ollama LLaVA model."""
    try:
        # Convert image to base64 if provided
        if image is not None:
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            
            # Save image to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Create Ollama request with image
            request_data = {
                "model": "llava",
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }
        else:
            # Text-only query
            request_data = {
                "model": "llava", 
                "prompt": prompt,
                "stream": False
            }
        
        # Call Ollama API
        result = subprocess.run([
            'curl', '-s', '-X', 'POST', 'http://localhost:11434/api/generate',
            '-H', 'Content-Type: application/json',
            '-d', json.dumps(request_data)
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response_data = json.loads(result.stdout)
            response = response_data.get('response', '').strip()
            print(f"[LLaVA-Ollama] Generated response: {response[:100]}..." if len(response) > 100 else f"[LLaVA-Ollama] Generated response: {response}")
            return response
        else:
            print(f"ERROR: Ollama request failed: {result.stderr}")
            return "error ollama"
            
    except Exception as e:
        print(f"ERROR: Ollama generation failed: {e}")
        return "error ollama"


if __name__ == '__main__':
    # Example usage and testing
    print("Initializing LLaVA model...")
    
    # Set the cache directory for model weights
    model_cache_dir = "/tmp/llava_model"
    
    try:
        initialize_llava_model(cache_dir=model_cache_dir)
        print("Model initialized successfully.")
        
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='blue')
        
        test_prompt = "What color is this image?"
        
        print(f"Testing LLaVA interface with prompt: '{test_prompt}'")
        
        generated_text = generate_text_from_image(test_image, test_prompt)
        
        print(f"Generated text: {generated_text}")
        
    except Exception as e:
        print(f"Test failed: {e}")