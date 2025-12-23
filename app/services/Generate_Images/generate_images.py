import os
import json
import requests
from dotenv import load_dotenv
from .generate_images_schema import GenerateImageResponse
from typing import Dict, Optional
from fastapi import UploadFile
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

class GenerateImages:
    def __init__(self):
        self.api_key = os.getenv("ARK_API_KEY")
        self.model = "seedream-4-0-250828"
        self.base_url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
        self.max_connections_for_parallel = 2  # If more connections than this, ignore them for parallel processing

    def generate_first_two_page(
        self,
        prompts: Dict[str, str],
        page_connections: Optional[Dict[str, str]],
        reference_image: UploadFile
    ) -> GenerateImageResponse:
        """Generate images for page 0 (cover) and page 1 only"""
        # Filter to only page 0 and page 1
        pages_to_generate = {k: v for k, v in prompts.items() if k in ['page 0', 'page 1']}
        
        image_urls = self._generate_images_for_pages(
            pages_to_generate,
            reference_image,
            page_connections=None,  # No connections for first two pages
            generated_images={}
        )
        
        return GenerateImageResponse(image_urls=image_urls)
    
    def generate_images(
        self,
        prompts: Dict[str, str],
        page_connections: Optional[Dict[str, str]],
        reference_image: UploadFile,
        coverpage: str = "no"
    ) -> GenerateImageResponse:
        """Generate images for all pages or skip cover/page 1 if they exist"""
        # Determine which pages to generate
        if coverpage.lower() == "yes":
            # Skip page 0 and page 1, generate pages 2-10
            pages_to_generate = {k: v for k, v in prompts.items() if k not in ['page 0', 'page 1']}
        else:
            # Generate all pages
            pages_to_generate = prompts
        
        # Decide whether to use page connections based on complexity
        # If too many connections, ignore them for parallel processing
        use_connections = False
        if page_connections and len(page_connections) <= self.max_connections_for_parallel:
            use_connections = True
        
        image_urls = self._generate_images_for_pages(
            pages_to_generate,
            reference_image,
            page_connections if use_connections else None,
            generated_images={}
        )
        
        return GenerateImageResponse(image_urls=image_urls)
    
    def _generate_images_for_pages(
        self,
        pages: Dict[str, str],
        reference_image: UploadFile,
        page_connections: Optional[Dict[str, str]],
        generated_images: Dict[str, str]
    ) -> Dict[str, str]:
        """Generate images for specified pages using SeeDream API"""
        image_urls = {}
        
        # Read reference image
        reference_image_bytes = reference_image.file.read()
        reference_image.file.seek(0)  # Reset file pointer
        
        if page_connections:
            # Sequential generation for pages with connections
            sorted_pages = sorted(pages.items(), key=lambda x: int(x[0].split()[1]))
            
            for page_key, prompt in sorted_pages:
                reference_page = None
                if page_key in page_connections:
                    ref_page_key = page_connections[page_key]
                    reference_page = generated_images.get(ref_page_key)
                
                image_url = self._generate_single_image(
                    prompt,
                    reference_image_bytes,
                    reference_page
                )
                
                image_urls[page_key] = image_url
                generated_images[page_key] = image_url
        else:
            # Parallel generation when no connections
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(
                        self._generate_single_image,
                        prompt,
                        reference_image_bytes,
                        None
                    ): page_key
                    for page_key, prompt in pages.items()
                }
                
                for future in futures:
                    page_key = futures[future]
                    try:
                        image_url = future.result()
                        image_urls[page_key] = image_url
                    except Exception as e:
                        print(f"Error generating image for {page_key}: {str(e)}")
                        raise Exception(f"Failed to generate image for {page_key}: {str(e)}")
        
        return image_urls
    
    def _generate_single_image(
        self,
        prompt: str,
        reference_image_bytes: bytes,
        reference_page_image: Optional[str] = None
    ) -> str:
        """Generate a single image using SeeDream API"""
        try:
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Encode reference image to base64
            import base64
            reference_image_base64 = base64.b64encode(reference_image_bytes).decode('utf-8')
            
            # Prepare payload
            payload = {
                'model': self.model,
                'prompt': prompt,
                'reference_image': reference_image_base64,
                'size': '1024x1024'
            }
            
            # If there's a reference page image, include it
            if reference_page_image:
                payload['reference_page_url'] = reference_page_image
            
            # Call SeeDream API
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract image URL from response (adjust based on actual API response format)
            image_url = result.get('data', [{}])[0].get('url') or result.get('image_url') or result.get('url')
            
            if not image_url:
                raise Exception(f"No image URL in response: {result}")
            
            return image_url
            
        except Exception as e:
            print(f"Error calling SeeDream API: {str(e)}")
            raise