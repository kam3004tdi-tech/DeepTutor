import os
from typing import Optional, Tuple, AsyncGenerator
from deeptutor.logging import get_logger

logger = get_logger("VertexAuth")

def log_to_file(msg: str):
    with open("/Users/kawasakimasanori/Desktop/VScode/DeepTutor/debug.log", "a") as f:
        f.write(msg + "\n")

def get_vertex_config() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get Vertex AI config from env.
    Returns (project_id, location, model_id).
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_REGION", "global")
    model_id = os.getenv("LLM_MODEL", "gemini-3.1-pro-preview")
    return project_id, location, model_id

async def vertex_complete(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
    **kwargs: object,
) -> str:
    """
    Complete using google-genai SDK on Vertex AI.
    """
    from google import genai
    from google.genai import types
    
    project_id, location, model_id = get_vertex_config()
    model_id = model or model_id
    
    log_to_file(f"[VertexAuth] Complete: model={model_id}, project={project_id}, location={location}")
    
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )
    
    safety_settings = [
        types.SafetySetting(category=cat, threshold="BLOCK_NONE")
        for cat in [
            "HARM_CATEGORY_HATE_SPEECH", 
            "HARM_CATEGORY_DANGEROUS_CONTENT", 
            "HARM_CATEGORY_HARASSMENT", 
            "HARM_CATEGORY_SEXUALLY_EXPLICIT"
        ]
    ]
    
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=kwargs.get("temperature", 0.7),
        max_output_tokens=kwargs.get("max_tokens", 4096),
        safety_settings=safety_settings,
    )
    
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config,
        )
        if not response.text:
            finish_reason = response.candidates[0].finish_reason if response.candidates else 'Unknown'
            log_to_file(f"[VertexAuth] Empty response. Reason: {finish_reason}")
        return response.text or ""
    except Exception as e:
        log_to_file(f"[VertexAuth] Complete failed: {e}")
        return ""

async def vertex_stream(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
    messages: list[dict[str, object]] | None = None,
    **kwargs: object,
) -> AsyncGenerator[str, None]:
    """
    Stream using google-genai SDK on Vertex AI.
    """
    from google import genai
    from google.genai import types
    
    project_id, location, model_id = get_vertex_config()
    model_id = model or model_id
    
    log_to_file(f"[VertexAuth] Stream Start: model={model_id}, project={project_id}, location={location}")
    
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )
    
    safety_settings = [
        types.SafetySetting(category=cat, threshold="BLOCK_NONE")
        for cat in [
            "HARM_CATEGORY_HATE_SPEECH", 
            "HARM_CATEGORY_DANGEROUS_CONTENT", 
            "HARM_CATEGORY_HARASSMENT", 
            "HARM_CATEGORY_SEXUALLY_EXPLICIT"
        ]
    ]
    
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=kwargs.get("temperature", 0.7),
        max_output_tokens=kwargs.get("max_tokens", 4096),
        safety_settings=safety_settings,
    )
    
    # Handle messages or prompt
    if messages:
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            if msg["role"] == "system":
                continue
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    else:
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    if not contents or (len(contents) == 1 and not contents[0].parts[0].text):
        log_to_file("[VertexAuth] Empty contents provided.")
        return

    try:
        log_to_file(f"[VertexAuth] Prompt preview: {contents[-1].parts[0].text[:50]}...")
        chunk_count = 0
        for chunk in client.models.generate_content_stream(
            model=model_id,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                chunk_count += 1
                yield chunk.text
            elif chunk.candidates and chunk.candidates[0].finish_reason not in {"STOP", None}:
                log_to_file(f"[VertexAuth] Stream finished unexpectedly. Reason: {chunk.candidates[0].finish_reason}")
        log_to_file(f"[VertexAuth] Stream finished. Total chunks: {chunk_count}")
    except Exception as e:
        log_to_file(f"[VertexAuth] Stream failed: {e}")
