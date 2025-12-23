from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.services.Generate_Story.generate_story_route import router as generate_story_router
from app.services.Generate_Images.generate_images_route import router as generate_images_router
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create necessary directories
os.makedirs("uploads/text_with_image", exist_ok=True)
os.makedirs("uploads/image_to_image", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Face2Toon AI API",
    description="API for story generation with content-based image descriptions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(generate_story_router, prefix="/api/v1")
app.include_router(generate_images_router, prefix="/api/v1")
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to BA CreateX API",
        "version": "1.0.0",
        "docs": "/docs",
        "services": {
            "text_with_image": "/api/v1/text-with-image",
            "image_to_image": "/api/v1/image-to-image"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.on_event("startup")
async def startup_event():
    """Startup event to check configuration"""
    # Check if API key is configured
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        print("Warning: ARK_API_KEY not found in environment variables")
    else:
        print("✅ API configuration loaded successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)