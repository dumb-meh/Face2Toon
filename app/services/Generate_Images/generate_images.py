import os
import json
import requests
from dotenv import load_dotenv
from .generate_images_schema import GenerateImageResponse
from typing import Dict, Optional
from fastapi import UploadFile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io
import base64
from urllib.parse import urljoin

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
        sequential: str = "no",
        story: Optional[Dict[str, str]] = None
    ) -> GenerateImageResponse:
        """Generate images for page 0 (cover) and page 1 only"""
        print(f"DEBUG generate_first_two_page: prompts type={type(prompts)}, page_connections type={type(page_connections)}")
        
        # Filter to only page 0 and page 1
        pages_to_generate = {k: v for k, v in prompts.items() if k in ['page 0', 'page 1']}
        story_to_generate = {k: v for k, v in story.items() if k in ['page 0', 'page 1']} if story else {}
        
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
            force_sequential=force_sequential,
            story=story_to_generate
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
        story: Optional[Dict[str, str]] = None
    ) -> GenerateImageResponse:
        """Generate images for all pages or skip cover/page 1 if they exist"""
        # Determine which pages to generate
        if coverpage.lower() == "yes":
            # Skip page 0 and page 1, generate pages 2-10
            pages_to_generate = {k: v for k, v in prompts.items() if k not in ['page 0', 'page 1']}
            story_to_generate = {k: v for k, v in story.items() if k not in ['page 0', 'page 1']} if story else {}
        else:
            # Generate all pages
            pages_to_generate = prompts
            story_to_generate = story if story else {}
        
        # Check if sequential generation is forced
        force_sequential = sequential.lower() == "yes"
        
        # For parallel mode, ignore page_connections completely
        # For sequential mode, use page_connections
        image_urls = self._generate_images_for_pages(
            pages_to_generate,
            reference_image,
            page_connections if force_sequential else None,
            gender=gender,
            age=age,
            image_style=image_style,
            force_sequential=force_sequential,
            story=story_to_generate
        )
        
        return GenerateImageResponse(image_urls=image_urls)
    
    def _generate_images_for_pages(
        self,
        pages: Dict[str, str],
        reference_image: UploadFile,
        page_connections: Optional[Dict[str, str]],
        gender: str,
        age: int,
        image_style: str,
        force_sequential: bool = False,
        story: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Generate images for specified pages using SeeDream API"""
        image_urls = {}
        story = story or {}
        
        # Read reference image
        reference_image_bytes = reference_image.file.read()
        reference_image.file.seek(0)  # Reset file pointer
        
        # If page_1_image is provided, save it for sequential generation starting from page 2
        page_1_bytes = None
        if page_1_image:
            page_1_bytes = page_1_image.file.read()
            page_1_image.file.seek(0)
        
        if force_sequential:
            # Sequential generation when forced
            sorted_pages = sorted(pages.items(), key=lambda x: int(x[0].split()[1]))
            
            for page_key, prompt in sorted_pages:
                reference_page = None
                use_raw_image = False
                use_page_1_image = False
                
                if page_key == "page 0":
                    # Only page 0 uses the raw reference image
                    use_raw_image = True
                else:
                    # For all other pages in sequential mode
                    page_num = int(page_key.split()[1])
                    prev_page_key = f"page {page_num - 1}"
                    
                    # Special case: page 2 should use page 1 image if provided
                    if page_key == "page 2" and page_1_bytes:
                        use_page_1_image = True
                    # Check if there's a specific page connection
                    elif page_connections and page_key in page_connections:
                        ref_page_key = page_connections[page_key]
                        reference_page = generated_images.get(ref_page_key)
                    
                    # If no specific connection, always use previous page for style consistency
                    if not reference_page and not use_page_1_image:
                        reference_page = generated_images.get(prev_page_key)
                    
                    # Don't use raw image for pages after page 0
                    use_raw_image = False
                
                # Debug logging
                print(f"Generating {page_key}:")
                print(f"  - Using raw image: {use_raw_image}")
                print(f"  - Using page 1 image: {use_page_1_image}")
                print(f"  - Reference page URL: {reference_page if reference_page else 'None'}")
                
                image_url = self._generate_single_image(
                    prompt,
                    reference_image_bytes if use_raw_image else (page_1_bytes if use_page_1_image else None),
                    reference_page,
                    gender,
                    age,
                    image_style,
                    page_key,
                    story.get(page_key)
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
                        page_key,
                        story.get(page_key)
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
        page_key: str,
        story_text: Optional[str] = None
    ) -> str:
        """Generate a single image using SeeDream API"""
        try:
            # Enhance the prompt with detailed style and character instructions
            if page_key == "page 0":
                # Cover page - include title rendering with text
                text_instruction = f"""
IMPORTANT: The image must include the following text rendered beautifully in the illustration:
Text to render: "{story_text}"
The text should be integrated into the cover design in an artistic, readable way suitable for a children's book title.
""" if story_text else """
Composition suitable for a book cover with space for title text.
"""
                enhanced_prompt = f"""
Children's storybook cover illustration in {image_style} style.
Main character: {age}-year-old {gender} child matching the reference image exactly.
{prompt}
{text_instruction}
Style: Professional children's book illustration, vibrant colors, high quality, storybook art, child-friendly, whimsical and engaging.
The child's face, features, hair, and appearance must exactly match the reference image provided.
""".strip()
            else:
                # Story pages - include story text to be rendered in the image
                text_instruction = f"""
IMPORTANT: The image must include the following story text rendered clearly and readably in the illustration:
Text to render: "{story_text}"
The text should be placed in a suitable location (top, bottom, or side) in a clear, readable font that complements the illustration style.
""" if story_text else ""
                enhanced_prompt = f"""
Children's storybook illustration in {image_style} style.
Main character: {age}-year-old {gender} child continuing from the previous page.
{prompt}
{text_instruction}
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
            
            # Encode reference image to base64
            reference_image_base64 = base64.b64encode(reference_image_bytes).decode('utf-8')
            
            # Prepare payload with 16:9 aspect ratio (will be resized to final dimensions)
            # Generate at 5120x2880 (16:9) to later resize to 5100x2550 (17"x8.5" at 300 DPI)
            # Note: Cannot use 'size' and 'width'/'height' together per SeeDream docs
            payload = {
                'model': self.model,
                'prompt': enhanced_prompt,
                'reference_image': reference_image_base64,
                'width': 5120,
                'height': 2880
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
            
            # Download and resize image to final dimensions
            resized_image_url = self._resize_image_to_print_size(image_url, page_key)
            
            return resized_image_url
            
        except Exception as e:
            print(f"Error calling SeeDream API: {str(e)}")
            raise
    
    def _resize_image_to_print_size(self, image_url: str, page_key: str) -> str:
        """Download image and resize to exact physical dimensions: 17" width x 8.5" height at 300 DPI"""
        try:
            # Download the image from Seedream
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            # Open image with PIL
            img = Image.open(io.BytesIO(response.content))
            
            # Physical dimensions in inches
            width_inches = 17.0
            height_inches = 8.5
            dpi = 300
            
            # Calculate pixel dimensions from physical size
            # 17 inches * 300 DPI = 5100 pixels width
            # 8.5 inches * 300 DPI = 2550 pixels height
            target_width = int(width_inches * dpi)
            target_height = int(height_inches * dpi)
            
            # Resize image to target dimensions
            # Using LANCZOS for high-quality resizing
            resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Save to uploads directory with proper DPI metadata
            os.makedirs('uploads/generated_images', exist_ok=True)
            page_num = page_key.replace('page ', '').replace(' ', '_')
            output_filename = f"uploads/generated_images/{page_num}_resized.png"
            
            # Save with DPI information embedded
            # This ensures the image will be 17" x 8.5" when printed or viewed in image software
            resized_img.save(output_filename, format='PNG', dpi=(dpi, dpi))
            
            print(f"Saved {page_key}: {target_width}x{target_height} pixels ({width_inches}\" x {height_inches}\" at {dpi} DPI)")
            
            # Construct public URL
            base_url = os.getenv('domain') or os.getenv('BASE_URL')
            # Ensure base_url doesn't end with slash
            base_url = base_url.rstrip('/')
            # Create public URL path
            public_url = f"{base_url}/{output_filename}"
            
            return public_url
            
        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            # Return original URL if resize fails
            return image_url