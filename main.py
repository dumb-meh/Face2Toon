from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.services.Generate_Story.generate_story_route import router as generate_story_router
from app.services.Generate_Images.generate_images_route import router as generate_images_router
from app.services.Regenerate_Image.regenerate_image_route import router as regenerate_images_router
from app.services.Create_Character.create_character_route import router as create_character_router
from app.services.Swap_Character.swap_character_route import router as swap_character_router
from app.services.Swap_Story_Information.swap_story_information_route import router as swap_story_information_router
# from test.test_generate_image import router as test_generate_batch_router
from app.services.Generate_Images_Batch.generate_images_batch_route import router as generate_images_batch_router
from app.utils import image_analysis as image_analysis_module
import os
import asyncio
from datetime import datetime, timedelta
import glob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create necessary directories
os.makedirs("uploads/text_with_image", exist_ok=True)
os.makedirs("uploads/image_to_image", exist_ok=True)
os.makedirs("uploads/generated_images", exist_ok=True)
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

# Mount static files directory to serve uploaded images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers
app.include_router(generate_story_router, prefix="/api/v1")
app.include_router(generate_images_router, prefix="/api/v1")
app.include_router(regenerate_images_router, prefix="/api/v1")
# Include image analysis router
app.include_router(image_analysis_module.router, prefix="/api/v1")
app.include_router(create_character_router, prefix="/api/v1")
app.include_router(swap_character_router, prefix="/api/v1")
app.include_router(swap_story_information_router, prefix="/api/v1")
# app.include_router(test_generate_batch_router, prefix="/api/v1", tags=["Test - Batch Generation"])
app.include_router(generate_images_batch_router, prefix="/api/v1", tags=["Batch Image Generation"])

async def cleanup_old_files():
    """Delete files older than 24 hours from uploads/generated_images"""
    while True:
        try:
            # Wait 1 hour between cleanup checks
            await asyncio.sleep(3600)
            
            # Get current time
            now = datetime.now()
            cutoff_time = now - timedelta(hours=24)
            
            # Find and delete old files
            pattern = "uploads/generated_images/*"
            files = glob.glob(pattern)
            
            deleted_count = 0
            for file_path in files:
                # Get file modification time
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Delete if older than 24 hours
                if file_mtime < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"Deleted old file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {str(e)}")
            
            if deleted_count > 0:
                print(f"Cleanup completed: {deleted_count} file(s) deleted")
                
        except Exception as e:
            print(f"Error in cleanup task: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Face2Toon API",
        "version": "1.0.0",
        "docs": "/docs",
        "services": {
            "generate_story": "/api/v1/generate-story",
            "generate_images": "/api/v1/generate-images",
            "image_analysis": "/api/v1/image-analysis"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.on_event("startup")
async def startup_event():
    """Startup event to check configuration and start cleanup task"""
    global cleanup_task
    
    # Check if API key is configured
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        print("Warning: ARK_API_KEY not found in environment variables")
    else:
        print("✅ API configuration loaded successfully")
    
    # Start cleanup background task
    cleanup_task = asyncio.create_task(cleanup_old_files())
    print("✅ File cleanup task started (deletes files older than 24 hours)")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event to cancel cleanup task"""
    global cleanup_task
    if cleanup_task:
        cleanup_task.cancel()
        print("🛑 File cleanup task stopped")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)