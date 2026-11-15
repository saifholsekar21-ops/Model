#!/usr/bin/env python3
"""Minimal text-generation API for a trained INFINITY AI Thinker adapter."""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import BitsAndBytesConfig, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

BASE_MODEL = os.getenv("INFINITY_BASE_MODEL", "Qwen/Qwen2.5-Omni-7B")
ADAPTER_PATH = os.getenv("INFINITY_ADAPTER_PATH", "outputs/infinity-ai-7b/adapter")
MAX_NEW_TOKENS = int(os.getenv("INFINITY_MAX_NEW_TOKENS", "512"))
runtime = {}

class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=2048)

class ChatResponse(BaseModel):
    model: str
    response: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required by this initial quantized deployment.")
    if not Path(ADAPTER_PATH).exists():
        raise RuntimeError(f"Adapter directory not found: {ADAPTER_PATH}")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    processor = Qwen2_5OmniProcessor.from_pretrained(BASE_MODEL)
    base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        BASE_MODEL, quantization_config=quant, device_map="auto",
        torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    runtime["processor"], runtime["model"] = processor, model
    yield
    runtime.clear()

app = FastAPI(title="INFINITY AI API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok" if runtime else "not_loaded", "model": BASE_MODEL}

@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not runtime:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    processor, model = runtime["processor"], runtime["model"]
    messages = [{"role": "user", "content": request.prompt}]
    inputs = processor.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
    try:
        with torch.inference_mode():
            output = model.generate(**inputs,
                max_new_tokens=request.max_new_tokens or MAX_NEW_TOKENS, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        answer = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return ChatResponse(model="INFINITY AI 1.0", response=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {type(exc).__name__}") from exc
