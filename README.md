# Name Origin Scoring LLM Service

This repository contains a FastAPI-based service that analyzes a given name and predicts the most likely countries of origin. Internally, it uses a locally hosted LLM (via [Ollama](https://github.com/jmorganca/ollama)) for name analysis and returns a JSON response with the predicted origins.

## Features

- **Name Analysis**: Receive a name through a GET endpoint (`/name_analyzer`) and get back up to 5 likely ISO alpha-3 country codes with confidence probabilities.
- **Local LLM Integration**: Uses a locally running LLM instance (e.g., `qwen2.5:7b`) loaded through Ollama.
- **ISO Validation**: Filters out invalid alpha-3 country codes using a local `iso_3166_countries.json` file.
- **Customizable**: Easily modify the prompt, max origins, or model temperature settings.

## Requirements

- **Python** 3.8+  
- **Ollama**: A local LLM server ([Ollama on GitHub](https://github.com/jmorganca/ollama)).
- **qwen2.5:7b Model**: Or any compatible model served by Ollama.

## Build the Docker Image
```bash
docker build -t name_analyzer_lastest .
```

## Run the service
```bash
docker run -it -p 8070:8070 name_analyzer_lastest:latest bash startup.sh
```

## Send a request
```bash
curl -X GET "http://localhost:8070/name_analyzer?name=Sara%20Abbas"
```

### Expected output
```bash
{"name":"Sara Abbas","likely_origins":[{"origin":"SAU","probability":0.45},{"origin":"IRQ","probability":0.25},{"origin":"PAK","probability":0.15},{"origin":"JOR","probability":0.1},{"origin":"BHR","probability":0.05}]}

```


## Code Explanation
### app.py (or main.py)

Creates a FastAPI app.
Defines the /name_analyzer endpoint which:
Validates input name.
Calls a locally hosted LLM for country-of-origin analysis.
Filters and validates ISO alpha-3 codes.
Returns exactly 5 origins, padding with "UNK" if needed.

### iso_3166_countries.json

A JSON file mapping alpha-3 codes to country names. Used to validate the codes from the LLM response.

### startup.sh

A script that starts Ollama, waits 5 seconds, then starts the uvicorn server.
