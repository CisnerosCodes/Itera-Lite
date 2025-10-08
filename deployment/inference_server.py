"""
FastAPI-based inference server for deployed Itera-Lite models.
Supports TorchScript, ONNX, and quantized models.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import logging
import json
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Itera-Lite Inference API",
    description="High-performance inference API for ultra-efficient Itera-Lite models",
    version="1.0.0"
)


class InferenceRequest(BaseModel):
    """Request model for inference."""
    text: str
    max_length: int = 50
    temperature: float = 1.0
    model_type: str = "torchscript"  # torchscript, onnx, quantized


class InferenceResponse(BaseModel):
    """Response model for inference."""
    generated_text: str
    input_length: int
    output_length: int
    latency_ms: float
    model_type: str


class ModelManager:
    """Manages loading and inference for multiple model formats."""
    
    def __init__(self, model_dir: str = "./deployment/models"):
        """
        Initialize model manager.
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.tokenizers = {}
        
        logger.info(f"Initializing ModelManager with directory: {model_dir}")
        self._load_available_models()
    
    def _load_available_models(self):
        """Load all available models from model directory."""
        if not self.model_dir.exists():
            logger.warning(f"Model directory {self.model_dir} does not exist")
            return
        
        # Load TorchScript models
        for pt_file in self.model_dir.glob("*_torchscript.pt"):
            model_name = pt_file.stem.replace("_torchscript", "")
            try:
                model = torch.jit.load(str(pt_file))
                model.eval()
                self.models[f"{model_name}_torchscript"] = model
                logger.info(f"✓ Loaded TorchScript model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load TorchScript model {model_name}: {e}")
        
        # Load ONNX models
        try:
            import onnxruntime as ort
            for onnx_file in self.model_dir.glob("*.onnx"):
                model_name = onnx_file.stem
                try:
                    session = ort.InferenceSession(str(onnx_file))
                    self.models[f"{model_name}_onnx"] = session
                    logger.info(f"✓ Loaded ONNX model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load ONNX model {model_name}: {e}")
        except ImportError:
            logger.warning("ONNX Runtime not available, skipping ONNX models")
        
        logger.info(f"Loaded {len(self.models)} models total")
    
    def inference_torchscript(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Run inference with TorchScript model.
        
        Args:
            model: TorchScript model
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs
        """
        with torch.no_grad():
            generated = input_ids.clone()
            
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                output = model(generated)
                
                # Handle tuple output
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token (assuming 0 is EOS)
                if next_token.item() == 0:
                    break
            
            return generated
    
    def inference_onnx(
        self,
        session,
        input_ids: np.ndarray,
        max_length: int = 50,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Run inference with ONNX model.
        
        Args:
            session: ONNX Runtime session
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs
        """
        input_name = session.get_inputs()[0].name
        generated = input_ids.copy()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            output = session.run(None, {input_name: generated})[0]
            
            # Get next token logits
            next_token_logits = output[:, -1, :] / temperature
            
            # Sample next token
            probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
            next_token = np.random.choice(len(probs[0]), p=probs[0])
            
            # Append to sequence
            generated = np.concatenate([generated, [[next_token]]], axis=1)
            
            # Stop if EOS token
            if next_token == 0:
                break
        
        return generated
    
    def predict(
        self,
        text: str,
        model_type: str = "torchscript",
        max_length: int = 50,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run inference on input text.
        
        Args:
            text: Input text
            model_type: Model type to use
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Prediction results
        """
        start_time = time.perf_counter()
        
        # Simple character-level tokenization for demo
        # In production, use proper tokenizer
        vocab = set(text)
        char_to_id = {ch: i for i, ch in enumerate(sorted(vocab))}
        char_to_id['<PAD>'] = len(char_to_id)
        char_to_id['<EOS>'] = len(char_to_id)
        id_to_char = {i: ch for ch, i in char_to_id.items()}
        
        # Tokenize input
        input_ids = [char_to_id.get(ch, char_to_id['<PAD>']) for ch in text]
        input_length = len(input_ids)
        
        # Find suitable model
        available_models = [k for k in self.models.keys() if model_type in k]
        if not available_models:
            raise ValueError(f"No {model_type} model available")
        
        model_key = available_models[0]
        model = self.models[model_key]
        
        # Run inference
        if "torchscript" in model_type:
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            output_tensor = self.inference_torchscript(
                model, input_tensor, max_length, temperature
            )
            output_ids = output_tensor[0].tolist()
        
        elif "onnx" in model_type:
            input_array = np.array([input_ids], dtype=np.int64)
            output_array = self.inference_onnx(
                model, input_array, max_length, temperature
            )
            output_ids = output_array[0].tolist()
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Decode output
        generated_text = ''.join([id_to_char.get(i, '?') for i in output_ids])
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'generated_text': generated_text,
            'input_length': input_length,
            'output_length': len(output_ids),
            'latency_ms': latency_ms,
            'model_type': model_type
        }


# Initialize model manager
model_manager = ModelManager()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Itera-Lite Inference API",
        "version": "1.0.0",
        "status": "running",
        "available_models": list(model_manager.models.keys())
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.models)
    }


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": list(model_manager.models.keys()),
        "count": len(model_manager.models)
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """
    Run inference on input text.
    
    Args:
        request: Inference request
        
    Returns:
        Inference response
    """
    try:
        result = model_manager.predict(
            text=request.text,
            model_type=request.model_type,
            max_length=request.max_length,
            temperature=request.temperature
        )
        return InferenceResponse(**result)
    
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark")
async def benchmark(
    text: str = "Hello world",
    model_type: str = "torchscript",
    num_runs: int = 10
):
    """
    Benchmark inference latency.
    
    Args:
        text: Input text
        model_type: Model type
        num_runs: Number of runs
        
    Returns:
        Benchmark results
    """
    try:
        latencies = []
        
        for _ in range(num_runs):
            result = model_manager.predict(
                text=text,
                model_type=model_type,
                max_length=20,
                temperature=1.0
            )
            latencies.append(result['latency_ms'])
        
        return {
            "model_type": model_type,
            "num_runs": num_runs,
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies)
        }
    
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
