from fastapi import FastAPI
from fastapi.exceptions import HTTPException

from typing import Optional, List
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
import json 
import os

# Set environment variable for GPU usage
os.environ["OLLAMA_USE_GPU"] = "1"

# Create an instance of the FastAPI application
app = FastAPI()

# Create an instance of an OpenAI client using a local endpoint (or whichever endpoint is specified).
# `instructor.from_openai` creates an 'instructor' client wrapping this OpenAI client.
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ),
    mode=instructor.Mode.JSON,
)

@app.get("/name_analyzer")
async def analyze_name(name: Optional[str] = None):
    """
    Endpoint to analyze a name and return a JSON with likely country origins (ISO codes) with confidence scores.
    Accepts a query parameter 'name'.
    Uses GPU acceleration for faster inference.
    """

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