from fastapi import FastAPI, BackgroundTasks
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
import json 
import os
import subprocess
import sys
import requests
import time

# Set environment variable for GPU usage
os.environ.update({
    "OLLAMA_USE_GPU": "1",
    "OLLAMA_GPU_LAYERS": "1000"
})

# Create an instance of the FastAPI application
app = FastAPI(
    title="Name Origin Analysis API",
    description="API for analyzing name origins using GPU-accelerated LLM inference",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create an instance of an OpenAI client using a local endpoint
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ),
    mode=instructor.Mode.JSON,
)

# Background task to monitor GPU usage during inference
def monitor_gpu_during_inference():
    """Monitor GPU usage during inference and log results"""
    try:
        # Initial GPU state
        initial = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        
        # Wait a bit during inference
        time.sleep(1)
        
        # Check GPU during inference
        during = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        
        # Log the results
        print(f"GPU MONITORING - Initial: {initial} | During inference: {during}", flush=True)
    except Exception as e:
        print(f"GPU monitoring error: {str(e)}", flush=True)

@app.get("/gpu_status")
async def check_gpu_status() -> Dict[str, Any]:
    """
    Endpoint to check the GPU status and verify if the application is using GPU acceleration.
    Returns information about GPU usage, Ollama configuration, and system details.
    """
    status_info = {
        "gpu_enabled": os.environ.get("OLLAMA_USE_GPU") == "1",
        "gpu_layers": os.environ.get("OLLAMA_GPU_LAYERS=1000", "Not set"),
        "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "Not specified"),
        "ollama_status": "Unknown",
        "gpu_stats": None,
        "system_info": {},
    }
    
    # Check if Ollama is running and using GPU
    try:
        # Test if Ollama API is accessible
        response = requests.get("http://localhost:11434/api/info")
        if response.status_code == 200:
            status_info["ollama_status"] = "Running"
            # Look for GPU indicators in the response
            if "gpu" in response.text.lower():
                status_info["ollama_status"] = "Running with GPU support"
            status_info["ollama_info"] = response.json()
    except Exception as e:
        status_info["ollama_status"] = f"Error: {str(e)}"
    
    # Try to get GPU stats using nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        if result.stdout:
            gpu_stats = []
            for i, line in enumerate(result.stdout.strip().split('\n')):
                values = [x.strip() for x in line.split(',')]
                if len(values) == 4:  # Make sure we have all 4 values
                    util, mem_used, mem_total, name = values
                    gpu_stats.append({
                        "gpu_id": i,
                        "name": name,
                        "utilization": util,
                        "memory_used": mem_used,
                        "memory_total": mem_total
                    })
            status_info["gpu_stats"] = gpu_stats
    except subprocess.CalledProcessError:
        status_info["gpu_stats"] = "nvidia-smi command failed"
    except FileNotFoundError:
        status_info["gpu_stats"] = "nvidia-smi not found, GPU might not be available"
    
    # Get Python and system info
    status_info["system_info"] = {
        "python_version": sys.version,
        "os_info": os.uname() if hasattr(os, 'uname') else "N/A"
    }
    
    # Check Ollama model status
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True, text=True, check=True
        )
        if result.stdout:
            status_info["ollama_models"] = result.stdout
    except Exception as e:
        status_info["ollama_models"] = f"Error getting model info: {str(e)}"
    
    return status_info

@app.get("/name_analyzer")
async def analyze_name(name: Optional[str] = None, background_tasks: BackgroundTasks = None):
    """
    Endpoint to analyze a name and return a JSON with likely country origins (ISO codes) with confidence scores.
    Accepts a query parameter 'name'.
    Uses GPU acceleration for faster inference.
    """
    # Start GPU monitoring in background
    if background_tasks:
        background_tasks.add_task(monitor_gpu_during_inference)

    # Validate that the name is provided and non-empty
    if not name or not name.strip():
        print("Received an empty name input", flush=True)
        raise HTTPException(
            status_code=400,
            detail="Name cannot be empty. Please provide a valid name."
        )
     
    # Log the received name
    print(f"Received name for analysis: {name}", flush=True)

    # Set the maximum number of origins to include in the response
    max_origins = 5

    # Load the dictionary of valid ISO alpha-3 codes from a local JSON file
    file_path='./data/iso_3166_countries.json'
    with open(file_path, 'r', encoding='utf-8') as file:
            alpha3 = json.load(file)

    try:
        # Force GPU usage for this request
        os.environ["OLLAMA_USE_GPU"] = "1"
        
        # Start time for performance measurement
        start_time = time.time()

        # Send the conversation prompt to the AI model
        # This prompt instructs the model how to analyze the name and format the JSON response.
        response = client.chat.completions.create(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze the following full name: "{name}"

    Consider the following factors:
    - Historical migration patterns
    - Linguistic and phonetic features 
    - Common variations of the given name
    - Family name origins and etymology
    - Cultural and regional naming conventions

    Your task:
    - Identify the top {max_origins} most likely countries of origin for the name provided.
    - Give all the origins as conventional alpha-3 ISO codes (all country ISO codes as described in the ISO 3166 international standard).
    - Assign a probability score to each country based on how likely the name originates from there.
    - Ensure the sum of all probabilities equals 1.0.
    - Ensure that the response is a JSON
    - Strictly follow this JSON structure in the response:
    {{
        "name": "<input_name>",
        "likely_origins": [
            {{"origin": <ISO_CODE>, "probability": <value>}},
            ...
        ]
    }}


    This is an example for the expected input and output: 
    Input:

    {{"name": "Hamza Fergougi" }}

    Output:

    {{
        "name": "Hamza Fergougi",
        "likely_origins": [
            {{
                "origin": "MAR",
                "probability": 0.85
            }},
            {{
                "origin": "DZA",
                "probability": 0.05
            }},
            {{
                "origin": "SYR",
                "probability": 0.04
            }},
            {{
                "origin": "BHR",
                "probability": 0.03
            }},
            {{
                "origin": "JOR",
                "probability": 0.03
            }}
        ]
    }}

    Ensure the following:
    - Every entry in 'likely_origins' has an 'origin' (ISO alpha-3 code) and a 'probability' (float).
    - The sum of probabilities equals 1.0.
    - Always return a list of exactly 5 items, even if some have a probability of 0.
    - Make sure to strictly return valid alpha-3 codes AND NOT the country names in the response.

    - Only output origins that are included in this dictionary.
    - Do not generate or infer any alpha-3 codes that are not explicitly listed in the dictionary.
    - Ensure all alpha-3 codes in the output match exactly with those provided in the dictionary.

                        
    """ }
            ],
            response_model=None,
            seed=42, # Setting the seed for reproducing the same output per query
            temperature=0.1, # Lower temperature to return a more deterministic response with less hallucinations
        )

        # Calculate processing time
        process_time = time.time() - start_time
        print(f"Name analysis completed in {process_time:.2f} seconds", flush=True)

        # Log the raw response
        raw_content = response.choices[0].message.content
        print(f"Raw response from AI: {raw_content}", flush=True)

        # Remove wrapping backticks or whitespace
        if raw_content.startswith("```json") and raw_content.endswith("```"):
            raw_content = raw_content[7:-3].strip()  # Remove ```json and ending ```

        # Validate raw response
        if not raw_content or not raw_content.startswith("{"):
            raise ValueError("Received invalid or empty response content.")

        # Parse the JSON content
        response_data = json.loads(raw_content)

        # Post-validation step to filter and validate alpha-3 codes
        valid_origins = set(alpha3.keys())  # Set of valid alpha-3 codes
        filtered_origins = []

        for origin in response_data["likely_origins"]:
            if origin["origin"] in valid_origins:
                filtered_origins.append(origin)
            else:
                print(f"Invalid alpha-3 code detected: {origin['origin']}", flush=True)

        # Ensure the response has exactly `max_origins` items by padding if necessary
        while len(filtered_origins) < max_origins:
            filtered_origins.append({"origin": "UNK", "probability": 0.0})  # Add padding with "UNK" or similar

        # Update the AI response with the filtered and padded origins
        response_data["likely_origins"] = filtered_origins[:max_origins]
        
        # Add processing time to response
        response_data["processing_time_seconds"] = process_time

        # Ensure each entry in 'likely_origins' is valid
        for origin in response_data["likely_origins"]:
            if "origin" not in origin or "probability" not in origin or not isinstance(origin["probability"], (float, int)):
                print("Invalid entry in 'likely_origins': missing 'origin' or 'probability'", flush=True)
                raise HTTPException(
                    status_code=500,
                    detail="AI response validation failed: 'likely_origins' contains invalid entries."
                )

        return response_data

    # Catch errors related to JSON parsing and raise an HTTP 500 error
    except json.JSONDecodeError as jde:
        print(f"JSON parsing error: {str(jde)}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: Invalid JSON format in AI response. {str(jde)}"
        )
    
    # Catch any general exceptions, log them, and raise an HTTP 500 error
    except Exception as e:
        print(f"Error during response validation: {str(e)}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Add an endpoint to force reload the model with GPU
@app.get("/reload_model")
async def reload_model_with_gpu():
    """Force reload the model with GPU settings"""
    try:
        # Kill existing Ollama process
        subprocess.run(["pkill", "ollama"], check=False)
        time.sleep(2)
        
        # Start Ollama with GPU
        subprocess.Popen(["ollama", "serve"], env={**os.environ, "OLLAMA_USE_GPU": "1", "OLLAMA_GPU_LAYERS": "1000"})
        time.sleep(5)
        
        # Pull model with GPU
        result = subprocess.run(
            ["ollama", "pull", "qwen2.5:7b"], 
            env={**os.environ, "OLLAMA_USE_GPU": "1"}, 
            capture_output=True, 
            text=True
        )
        
        # Check model status
        status = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
        
        return {
            "reload_status": "success",
            "model_status": status,
            "gpu_enabled": os.environ.get("OLLAMA_USE_GPU") == "1",
            "gpu_layers": os.environ.get("OLLAMA_GPU_LAYERS", "Not set"),
            "pull_result": result.stdout
        }
    except Exception as e:
        return {
            "reload_status": "error",
            "message": str(e)
        }