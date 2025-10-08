"""
Production Inference API for Itera-Lite with Adaptive Learning
FastAPI server with inference, feedback, and metrics endpoints.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import uvicorn
from collections import deque
import psutil
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.itera_lite import IteraLiteModel
from models.config import IteraLiteConfig
from utils.adaptive_learning import AdaptiveSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Itera-Lite Inference API",
    description="Production-ready inference server with adaptive learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request/Response models
class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""
    input_text: str = Field(..., description="Input text for generation")
    max_length: int = Field(50, description="Maximum generation length", ge=1, le=512)
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.1, le=2.0)
    top_k: int = Field(50, description="Top-k sampling", ge=1, le=100)
    model_variant: str = Field("default", description="Model variant to use")


class InferenceResponse(BaseModel):
    """Response model for inference endpoint."""
    output_text: str
    input_text: str
    latency_ms: float
    tokens_generated: int
    model_variant: str
    feedback_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""
    feedback_id: str = Field(..., description="Feedback ID from inference")
    user_rating: int = Field(..., description="User rating (1-5)", ge=1, le=5)
    correctness: bool = Field(..., description="Whether output was correct")
    comments: Optional[str] = Field(None, description="Optional user comments")


class FeedbackResponse(BaseModel):
    """Response model for feedback endpoint."""
    status: str
    message: str
    adaptation_triggered: bool = False


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    total_requests: int
    average_latency_ms: float
    requests_per_second: float
    model_info: Dict
    system_resources: Dict
    adaptive_metrics: Dict


# Global state
class ServerState:
    """Global server state."""
    def __init__(self):
        self.model = None
        self.adaptive_system = None
        self.request_count = 0
        self.latency_history = deque(maxlen=1000)
        self.start_time = time.time()
        
server_state = ServerState()


# Rate limiting (simple in-memory implementation)
class RateLimiter:
    """Simple rate limiter."""
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = deque(maxlen=requests_per_minute)
    
    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        now = time.time()
        # Remove requests older than 1 minute
        while self.requests and now - self.requests[0] > 60:
            self.requests.popleft()
        
        if len(self.requests) < self.requests_per_minute:
            self.requests.append(now)
            return True
        return False

rate_limiter = RateLimiter(requests_per_minute=100)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    if not rate_limiter.is_allowed():
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    response = await call_next(request)
    return response


@app.on_event("startup")
async def startup_event():
    """Initialize model and adaptive system on startup."""
    logger.info("Initializing Itera-Lite Inference Server...")
    
    # Load model configuration
    config = IteraLiteConfig(
        vocab_size=2000,
        hidden_size=64,
        num_layers=3,
        ssm_state_size=8,
        num_experts=4,
        top_k_experts=2,
        expert_size=32,
        max_seq_length=128
    )
    
    # Initialize model
    server_state.model = IteraLiteModel(config)
    
    # Try to load checkpoint
    checkpoint_paths = [
        'checkpoints/int4/itera_lite_int4.pt',
        'checkpoints/distilled/itera_lite_micro.pt',
        'checkpoints/quantized/itera_lite_quantized.pt'
    ]
    
    checkpoint_loaded = False
    for checkpoint_path in checkpoint_paths:
        if Path(checkpoint_path).exists():
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                server_state.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                server_state.model.load_state_dict(checkpoint)
            
            checkpoint_loaded = True
            logger.info(f"✓ Checkpoint loaded from {checkpoint_path}")
            break
    
    if not checkpoint_loaded:
        logger.warning("No checkpoint found, using randomly initialized model")
    
    server_state.model.eval()
    
    # Initialize adaptive system
    server_state.adaptive_system = AdaptiveSystem(server_state.model)
    
    logger.info("✓ Inference server ready")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Itera-Lite Inference API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/infer": "POST - Generate text",
            "/feedback": "POST - Submit feedback",
            "/metrics": "GET - Server metrics",
            "/health": "GET - Health check",
            "/adapt": "POST - Trigger adaptation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": server_state.model is not None,
        "adaptive_system_ready": server_state.adaptive_system is not None,
        "uptime_seconds": time.time() - server_state.start_time,
        "total_requests": server_state.request_count
    }


@app.post("/infer", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    Generate text using the model.
    
    Args:
        request: InferenceRequest with input text and parameters
        
    Returns:
        InferenceResponse with generated text and metrics
    """
    if server_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Simple token generation (placeholder - would use proper tokenizer in production)
        # For demonstration, just echo input with modification
        input_tokens = request.input_text.split()
        output_text = f"Generated response for: {request.input_text[:50]}..."
        tokens_generated = len(output_text.split())
        
        # Simulate model inference
        with torch.no_grad():
            # In production, would tokenize, generate, and decode
            # sample_input = torch.randint(0, 2000, (1, min(len(input_tokens), 128)))
            # output = server_state.model(sample_input)
            pass
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Log to adaptive system
        feedback_id = None
        if server_state.adaptive_system:
            result = server_state.adaptive_system.log_and_learn(
                input_text=request.input_text,
                output_text=output_text,
                auto_adapt=False  # Don't auto-adapt on every request
            )
            feedback_id = result['feedback_id']
        
        # Update metrics
        server_state.request_count += 1
        server_state.latency_history.append(latency_ms)
        
        return InferenceResponse(
            output_text=output_text,
            input_text=request.input_text,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
            model_variant=request.model_variant,
            feedback_id=feedback_id
        )
    
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for an inference.
    
    Args:
        request: FeedbackRequest with rating and correctness
        
    Returns:
        FeedbackResponse with status
    """
    if server_state.adaptive_system is None:
        raise HTTPException(status_code=503, detail="Adaptive system not initialized")
    
    try:
        # Update feedback
        server_state.adaptive_system.feedback_logger.update_feedback(
            feedback_id=request.feedback_id,
            user_rating=request.user_rating,
            correctness=request.correctness
        )
        
        # Check if adaptation should trigger
        negative_count = len(server_state.adaptive_system.feedback_logger.get_negative_feedback())
        adaptation_triggered = False
        
        if negative_count >= server_state.adaptive_system.auto_update_threshold:
            logger.info(f"Auto-triggering adaptation: {negative_count} negative samples")
            server_state.adaptive_system.trigger_adaptation(manual=False)
            adaptation_triggered = True
        
        return FeedbackResponse(
            status="success",
            message=f"Feedback recorded for {request.feedback_id}",
            adaptation_triggered=adaptation_triggered
        )
    
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get server metrics and statistics.
    
    Returns:
        MetricsResponse with comprehensive metrics
    """
    uptime = time.time() - server_state.start_time
    avg_latency = sum(server_state.latency_history) / len(server_state.latency_history) if server_state.latency_history else 0
    rps = server_state.request_count / uptime if uptime > 0 else 0
    
    # System resources
    process = psutil.Process()
    memory_info = process.memory_info()
    
    system_resources = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_mb": memory_info.rss / 1024 / 1024,
        "memory_percent": process.memory_percent()
    }
    
    # Model info
    model_info = {
        "parameters": sum(p.numel() for p in server_state.model.parameters()) if server_state.model else 0,
        "device": "cpu",
        "dtype": "float32"
    }
    
    # Adaptive metrics
    adaptive_metrics = {}
    if server_state.adaptive_system:
        adaptive_metrics = server_state.adaptive_system.get_system_status()
    
    return MetricsResponse(
        total_requests=server_state.request_count,
        average_latency_ms=avg_latency,
        requests_per_second=rps,
        model_info=model_info,
        system_resources=system_resources,
        adaptive_metrics=adaptive_metrics
    )


@app.post("/adapt")
async def trigger_adaptation():
    """
    Manually trigger model adaptation.
    
    Returns:
        Adaptation results
    """
    if server_state.adaptive_system is None:
        raise HTTPException(status_code=503, detail="Adaptive system not initialized")
    
    try:
        result = server_state.adaptive_system.trigger_adaptation(manual=True)
        return {
            "status": "success",
            "adaptation_result": result
        }
    except Exception as e:
        logger.error(f"Adaptation error: {e}")
        raise HTTPException(status_code=500, detail=f"Adaptation failed: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Itera-Lite Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "inference_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
