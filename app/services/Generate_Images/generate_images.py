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
        self.parallel_batch_size = 5  # Generate 5 images at once in parallel mode

    def generate_first_two_page(
        self,
        prompts: Dict[str, str],
        page_connections: Optional[Dict[str, str]],
        reference_image: UploadFile,
        gender: str,
        age: int,
        image_style: str,
        sequential: str = "no"
    ) -> GenerateImageResponse:
        """Generate images for page 0 (cover) and page 1 only"""
        # Filter to only page 0 and page 1
        pages_to_generate = {k: v for k, v in prompts.items() if k in ['page 0', 'page 1']}
        
        # Force sequential if requested
        force_sequential = sequential.lower() == "yes"
        
        image_urls = self._generate_images_for_pages(
            pages_to_generate,
            reference_image,
            page_connections=page_connections if force_sequential else None,
            generated_images={},
            gender=gender,
            age=age,
            image_style=image_style,
            force_sequential=force_sequential
        )
        
        return GenerateImageResponse(image_urls=image_urls)
    
    def generate_images(
        self,
        prompts: Dict[str, str],
        page_connections: Optional[Dict[str, str]],
        reference_image: UploadFile,
        gender: str,
        age: int,
        image_style: str,
        coverpage: str = "no",
        sequential: str = "no",
        existing_pages: Dict[str, str] = None
    ) -> GenerateImageResponse:
        """Generate images for all pages or skip cover/page 1 if they exist"""
        # Determine which pages to generate
        if coverpage.lower() == "yes":
            # Skip page 0 and page 1, generate pages 2-10
            pages_to_generate = {k: v for k, v in prompts.items() if k not in ['page 0', 'page 1']}
        else:
            # Generate all pages
            pages_to_generate = prompts
        
        # Check if sequential generation is forced
        force_sequential = sequential.lower() == "yes"
        
        # Use existing pages if provided, otherwise start with empty dict
        initial_generated_images = existing_pages if existing_pages else {}
        
        # For parallel mode, ignore page_connections completely
        # For sequential mode, use page_connections
        image_urls = self._generate_images_for_pages(
            pages_to_generate,
            reference_image,
            page_connections if force_sequential else None,
            generated_images=initial_generated_images,
            gender=gender,
            age=age,
            image_style=image_style,
            force_sequential=force_sequential
        )
        
        return GenerateImageResponse(image_urls=image_urls)
    
    def _generate_images_for_pages(
        self,
        pages: Dict[str, str],
        reference_image: UploadFile,
        page_connections: Optional[Dict[str, str]],
        generated_images: Dict[str, str],
        gender: str,
        age: int,
        image_style: str,
        force_sequential: bool = False
    ) -> Dict[str, str]:
        """Generate images for specified pages using SeeDream API"""
        image_urls = {}
        
        # Read reference image
        reference_image_bytes = reference_image.file.read()
        reference_image.file.seek(0)  # Reset file pointer
        
        if page_connections or force_sequential:
            # Sequential generation for pages with connections or when forced
            sorted_pages = sorted(pages.items(), key=lambda x: int(x[0].split()[1]))
            
            for page_key, prompt in sorted_pages:
                reference_page = None
                use_raw_image = False
                
                if page_key == "page 0":
                    # Only page 0 uses the raw reference image
                    use_raw_image = True
                else:
                    # For all other pages in sequential mode
                    page_num = int(page_key.split()[1])
                    prev_page_key = f"page {page_num - 1}"
                    
                    # Check if there's a specific page connection
                    if page_connections and page_key in page_connections:
                        ref_page_key = page_connections[page_key]
                        reference_page = generated_images.get(ref_page_key)
                    
                    # If no specific connection, always use previous page for style consistency
                    if not reference_page:
                        reference_page = generated_images.get(prev_page_key)
                    
                    # Don't use raw image for pages after page 0
                    use_raw_image = False
                
                # Debug logging
                print(f"Generating {page_key}:")
                print(f"  - Using raw image: {use_raw_image}")
                print(f"  - Reference page URL: {reference_page if reference_page else 'None'}")
                
                image_url = self._generate_single_image(
                    prompt,
                    reference_image_bytes if use_raw_image else None,
                    reference_page,
                    gender,
                    age,
                    image_style,
                    page_key
                )
                
                image_urls[page_key] = image_url
                generated_images[page_key] = image_url
                print(f"  - Generated URL: {image_url}")
            
            # Clear generated_images dict to free memory after sequential generation
            generated_images.clear()
            print("Cleared generated_images cache after sequential generation")
        else:
            # Parallel generation - all pages use the reference image
            # Generate 5 images at once to optimize performance
            print(f"Parallel generation mode: Generating {len(pages)} pages in batches of {self.parallel_batch_size}\"")
            print(f"All pages will use the reference image, ignoring any page connections")
            
            with ThreadPoolExecutor(max_workers=self.parallel_batch_size) as executor:
                futures = {
                    executor.submit(
                        self._generate_single_image,
                        prompt,
                        reference_image_bytes,  # All pages get reference image in parallel mode
                        None,  # No previous page reference in parallel mode
                        gender,
                        age,
                        image_style,
                        page_key
                    ): page_key
                    for page_key, prompt in pages.items()
                }
                
                for future in futures:
                    page_key = futures[future]
                    try:
                        image_url = future.result()
                        image_urls[page_key] = image_url
                        print(f"  - {page_key}: Generated successfully")
                    except Exception as e:
                        print(f"Error generating image for {page_key}: {str(e)}")
                        raise Exception(f"Failed to generate image for {page_key}: {str(e)}")
        
        return image_urls
    
    def _generate_single_image(
        self,
        prompt: str,
        reference_image_bytes: Optional[bytes],
        reference_page_image: Optional[str],
        gender: str,
        age: int,
        image_style: str,
        page_key: str
    ) -> str:
        """Generate a single image using SeeDream API"""
        try:
            # Enhance the prompt with detailed style and character instructions
            if page_key == "page 0":
                # Cover page - uses the original uploaded reference image
                enhanced_prompt = f"""
Children's storybook cover illustration in {image_style} style.
Main character: {age}-year-old {gender} child matching the reference image exactly.
{prompt}
Style: Professional children's book illustration, vibrant colors, high quality, storybook art, child-friendly, whimsical and engaging.
IMPORTANT: The child's face, facial features, eyebrows, eye shape, nose, mouth, hair color, hair style, and skin tone must EXACTLY match the reference image. Do not alter or reinterpret any facial characteristics.
Composition suitable for a book cover with space for title text.
Negative prompt: Do not change face structure, facial proportions, eyebrow shape, eye color, hair style, or skin tone. No artistic reinterpretation of facial features.
""".strip()
            else:
                # Story pages - maintain consistency from previous page's generated image
                enhanced_prompt = f"""
Children's storybook illustration in {image_style} style.
Main character: {age}-year-old {gender} child continuing from the previous page.
{prompt}
Style: Professional children's book illustration, vibrant colors, high quality, storybook art, child-friendly, whimsical and engaging.
CRITICAL: Maintain EXACT character appearance from the style reference - same facial features, eyebrows, eye shape, nose, mouth, hair color, hair style, and skin tone. If clothing colors or patterns are specified in the prompt, follow them precisely without variation.
Focus on: implementing the exact scene, actions, setting, and clothing details as described while preserving all character appearance characteristics.
Negative prompt: No changes to the character's face structure, facial proportions, eyebrow thickness or shape, eye color or shape, nose shape, mouth shape, hair color, hair style, or skin tone. Do not modify clothing colors from the prompt description. No artistic reinterpretation of the character's established appearance.
""".strip()
            
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Prepare payload
            payload = {
                'model': self.model,
                'prompt': enhanced_prompt,
                'size': '1024x1024'
            }
            
            # Add reference image only if provided (page 0 only in sequential mode)
            if reference_image_bytes:
                import base64
                reference_image_base64 = base64.b64encode(reference_image_bytes).decode('utf-8')
                payload['reference_image'] = reference_image_base64
            
            # If there's a reference page image (for sequential generation), include it as style reference
            # This ensures the model uses the previous page's character appearance and style
            if reference_page_image:
                payload['style_reference_url'] = reference_page_image
            
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