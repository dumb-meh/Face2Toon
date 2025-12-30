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
import uuid
from datetime import datetime

load_dotenv()

class GenerateImages:
    def __init__(self):
        self.api_key = os.getenv("ARK_API_KEY")
        self.model = "seedream-4-5-251128"
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
        
        # Generate unique session ID for this request
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Filter to only page 0 and page 1
        pages_to_generate = {k: v for k, v in prompts.items() if k in ['page 0', 'page 1']}
        story_to_generate = {k: v for k, v in story.items() if k in ['page 0', 'page 1']} if story else {}
        
        # Force sequential if requested
        force_sequential = sequential.lower() == "yes"
        
        image_urls = self._generate_images_for_pages(
            pages_to_generate,
            reference_image,
            page_connections=page_connections if force_sequential else None,
            gender=gender,
            age=age,
            image_style=image_style,
            force_sequential=force_sequential,
            story=story_to_generate,
            session_id=session_id,
            should_split=False  # Don't split for first two pages
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
        story: Optional[Dict[str, str]] = None,
        page_1_url: Optional[str] = None
    ) -> GenerateImageResponse:
        """Generate images for all pages or skip cover/page 1 if they exist"""
        # Generate unique session ID for this request
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Determine which pages to generate
        if coverpage.lower() == "yes":
            # Skip page 0 and page 1, generate pages 2-10
            pages_to_generate = {k: v for k, v in prompts.items() if k not in ['page 0', 'page 1']}
            story_to_generate = {k: v for k, v in story.items() if k not in ['page 0', 'page 1']} if story else {}
        else:
            # Generate all pages, but if page_1_url is provided, skip page 1 since it's already available
            if page_1_url:
                pages_to_generate = {k: v for k, v in prompts.items() if k != 'page 1'}
                story_to_generate = {k: v for k, v in story.items() if k != 'page 1'} if story else {}
            else:
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
            story=story_to_generate,
            session_id=session_id,
            page_1_url=page_1_url,
            should_split=True  # Enable splitting for full generation
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
        story: Optional[Dict[str, str]] = None,
        session_id: str = None,
        page_1_url: Optional[str] = None,
        should_split: bool = False
    ) -> Dict[str, str]:
        """Generate images for specified pages using SeeDream API"""
        image_urls = {}
        generated_images = {}  # Initialize for sequential generation
        generated_images_bytes = {}  # Store bytes for direct use
        story = story or {}
        
        # Determine starting page counter based on whether page_1_url is provided
        # If page_1_url is provided and will be split into pages 1-2, next generated image starts at image 1 (pages 3-4)
        # If no page_1_url, first generated image is image 0 for page 0, or image 1 for pages 1-2 (if skipping cover)
        page_counter_start = 0
        
        # Read reference image
        reference_image_bytes = reference_image.file.read()
        reference_image.file.seek(0)  # Reset file pointer
        
        # If page_1_url is provided, read the image from disk and split it
        if page_1_url and should_split:
            try:
                # Extract file path from URL
                if page_1_url.startswith('http'):
                    from urllib.parse import urlparse
                    parsed = urlparse(page_1_url)
                    file_path = parsed.path.lstrip('/')
                else:
                    file_path = page_1_url.lstrip('/')
                
                # Read the image file directly from disk
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        page_1_bytes = f.read()
                    # Store bytes for AI use
                    generated_images_bytes['page 1'] = page_1_bytes
                    
                    # Split page 1 image into page 1 and page 2
                    img = Image.open(io.BytesIO(page_1_bytes))
                    middle_x = img.width // 2
                    
                    left_half = img.crop((0, 0, middle_x, img.height))
                    right_half = img.crop((middle_x, 0, img.width, img.height))
                    
                    os.makedirs('uploads/generated_images/splitted', exist_ok=True)
                    left_filename = f"uploads/generated_images/splitted/{session_id}_page_1.png"
                    right_filename = f"uploads/generated_images/splitted/{session_id}_page_2.png"
                    
                    left_half.save(left_filename, format='PNG', dpi=(300, 300))
                    right_half.save(right_filename, format='PNG', dpi=(300, 300))
                    
                    base_url = os.getenv('domain') or os.getenv('BASE_URL')
                    base_url = base_url.rstrip('/')
                    
                    image_urls['page 1'] = f"{base_url}/{left_filename}"
                    image_urls['page 2'] = f"{base_url}/{right_filename}"
                    
                    print(f"Split provided page 1 into page 1 and page 2")
                    print(f"  - page 1: {image_urls['page 1']}")
                    print(f"  - page 2: {image_urls['page 2']}")
                    
                    page_counter_start = 1  # Next generated image will be image 1 (pages 3-4)
                else:
                    print(f"Warning: Page 1 image not found at: {file_path}")
            except Exception as e:
                print(f"Error loading/splitting page 1 image: {str(e)}")
        
        page_counter = page_counter_start
        
        if force_sequential:
            # Sequential generation when forced
            sorted_pages = sorted(pages.items(), key=lambda x: int(x[0].split()[1]))
            
            for page_key, prompt in sorted_pages:
                reference_page = None
                reference_page_bytes = None
                use_raw_image = False
                
                # Special handling for page 0 (cover)
                if page_key == "page 0":
                    # Only page 0 uses the raw reference image
                    use_raw_image = True
                    # Page 0 always gets page_number=0 and should NEVER be split
                    current_page_number = 0
                else:
                    # For all other pages in sequential mode
                    page_num = int(page_key.split()[1])
                    prev_page_key = f"page {page_num - 1}"
                    
                    # Check if there's a specific page connection
                    if page_connections and page_key in page_connections:
                        ref_page_key = page_connections[page_key]
                        # Try to get bytes first, then URL
                        reference_page_bytes = generated_images_bytes.get(ref_page_key)
                        if not reference_page_bytes:
                            reference_page = generated_images.get(ref_page_key)
                    
                    # If no specific connection, always use previous page for style consistency
                    if not reference_page and not reference_page_bytes:
                        reference_page_bytes = generated_images_bytes.get(prev_page_key)
                        if not reference_page_bytes:
                            reference_page = generated_images.get(prev_page_key)
                    
                    # Don't use raw image for pages after page 0
                    use_raw_image = False
                    # Use page_counter for non-cover pages
                    current_page_number = page_counter
                
                # Debug logging
                print(f"Generating {page_key}:")
                print(f"  - Using raw image: {use_raw_image}")
                print(f"  - Reference page bytes: {len(reference_page_bytes) if reference_page_bytes else 0} bytes")
                print(f"  - Reference page URL: {reference_page if reference_page else 'None'}")
                print(f"  - Page number for splitting: {current_page_number}")
                
                image_urls_dict = self._generate_single_image(
                    prompt,
                    reference_image_bytes if use_raw_image else None,
                    reference_page,
                    gender,
                    age,
                    image_style,
                    page_key,
                    story.get(page_key),
                    session_id,
                    reference_page_bytes=reference_page_bytes,
                    should_split=should_split,
                    page_number=current_page_number
                )
                
                # Store full URL for AI reference in next pages
                # For pages with custom keys (coloring pages), get the actual URL value
                if 'full' in image_urls_dict:
                    generated_images[page_key] = image_urls_dict['full']
                    print(f"  - Generated full URL: {image_urls_dict['full']}")
                else:
                    # Get the first URL value (for coloring pages with custom keys)
                    url_value = next(iter(image_urls_dict.values()))
                    generated_images[page_key] = url_value
                    print(f"  - Generated URL: {url_value}")
                
                # Add URLs to response based on whether page is split
                if should_split and page_key != 'page 0':
                    # Check if page was actually split or returned with a different key
                    page_num = int(page_key.split()[1]) if page_key.startswith('page ') else 0
                    is_coloring_page = page_num == 12 or page_num == 13
                    
                    if is_coloring_page:
                        # Coloring pages are not split, add with their custom key (page 23, 24)
                        for key, url in image_urls_dict.items():
                            if key != 'full':
                                image_urls[key] = url
                                print(f"  - {key}: {url}")
                    else:
                        # Split pages - add individual page URLs
                        for key, url in image_urls_dict.items():
                            if key != 'full':
                                image_urls[key] = url
                                print(f"  - {key}: {url}")
                        page_counter += 1  # Increment for next image
                else:
                    # No splitting OR cover page - return full URL with original key
                    image_urls[page_key] = image_urls_dict['full']
                    # Only increment page_counter if this is NOT page 0
                    if page_key != 'page 0':
                        page_counter += 1
            
            # Clear generated_images dict to free memory after sequential generation
            generated_images.clear()
            print("Cleared generated_images cache after sequential generation")
        else:
            # Parallel generation - all pages use the reference image
            # Generate 5 images at once to optimize performance
            print(f"Parallel generation mode: Generating {len(pages)} pages in batches of {self.parallel_batch_size}")
            print(f"All pages will use the reference image, ignoring any page connections")
            
            with ThreadPoolExecutor(max_workers=self.parallel_batch_size) as executor:
                futures = {}
                current_page_counter = page_counter
                for page_key, prompt in pages.items():
                    # Determine page_number for splitting calculation
                    # Page 0 (cover) always uses 0 but won't be split
                    # First splittable page (page 1) should use 0 to become pages 1-2
                    if page_key == 'page 0':
                        page_num_for_split = 0
                    else:
                        page_num_for_split = current_page_counter
                        current_page_counter += 1
                    
                    future = executor.submit(
                        self._generate_single_image,
                        prompt,
                        reference_image_bytes,  # All pages get reference image in parallel mode
                        None,  # No previous page reference in parallel mode
                        gender,
                        age,
                        image_style,
                        page_key,
                        story.get(page_key),
                        session_id,
                        should_split=should_split,
                        page_number=page_num_for_split
                    )
                    futures[future] = page_key
                
                for future in futures:
                    page_key = futures[future]
                    try:
                        image_urls_dict = future.result()
                        
                        # Check if this page was actually split (has keys other than 'full')
                        has_split_pages = any(key != 'full' for key in image_urls_dict.keys())
                        
                        if has_split_pages:
                            # Page was split - add split page URLs
                            for key, url in image_urls_dict.items():
                                if key != 'full':
                                    image_urls[key] = url
                                    print(f"  - {key}: {url}")
                        else:
                            # Page was not split (page 0, coloring pages, or splitting disabled)
                            # Check if dict has a specific key (like 'page 23' for coloring pages)
                            for key, url in image_urls_dict.items():
                                if key != 'full':
                                    # Use the specific key (e.g., 'page 23', 'page 24')
                                    image_urls[key] = url
                                    print(f"  - {key}: {url}")
                                else:
                                    # No specific key, use original page_key
                                    image_urls[page_key] = url
                                    print(f"  - {page_key}: {url}")
                    except Exception as e:
                        print(f"Error generating image for {page_key}: {str(e)}")
                        raise Exception(f"Failed to generate image for {page_key}: {str(e)}")
        
        print(f"\nFinal image_urls dictionary contains {len(image_urls)} pages:")
        for key in sorted(image_urls.keys(), key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else 0):
            print(f"  - {key}")
        
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
        story_text: Optional[str] = None,
        session_id: str = None,
        reference_page_bytes: Optional[bytes] = None,
        should_split: bool = False,
        page_number: int = None
    ) -> Dict[str, str]:
        """Generate a single image using SeeDream API"""
        try:
            # Enhance the prompt with detailed style and character instructions
            if page_key == "page 0":
                # Cover page - include title rendering with text
                text_instruction = f"""
CRITICAL TEXT RENDERING REQUIREMENT:
You MUST include the COMPLETE title text exactly as written below in the generated image.

FULL TITLE TO RENDER:
"{story_text}"

TEXT PLACEMENT INSTRUCTIONS:
- Integrate the title prominently into the cover design
- Use an artistic, readable font suitable for a children's book title
- Ensure the ENTIRE title is visible and readable
- The text should be beautifully styled but COMPLETE from start to finish
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
                # Check if this is a coloring page (pages 12 or 13)
                page_num = int(page_key.split()[1]) if page_key.startswith('page ') else 0
                is_coloring_page = page_num == 12 or page_num == 13
                
                # Story pages - include story text to be rendered in the image (except coloring pages)
                if is_coloring_page:
                    # No text for coloring pages
                    text_instruction = ""
                else:
                    # Count words in the story text to emphasize completeness
                    word_count = len(story_text.split()) if story_text else 0
                    final_word = story_text.split()[-1] if story_text else ''
                    text_instruction = f"""
COMPLETE TEXT TO RENDER IN IMAGE (every single word required):

{story_text}

TEXT MUST END WITH: "{final_word}"

Requirements:
- Use Comic Relief font style (playful, rounded)
- Reduce font size if necessary to fit ALL {word_count} words
- Use as many lines as needed to show the complete text
- Place text on left or right side where there's open space
- Do not cover character faces
- Simple text overlay - no frames or boxes
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
            
            # Prepare payload with appropriate dimensions
            # Pages 0, 12, 13: 8.5"x8.5" at 300 DPI = 2550x2550 pixels
            # Other pages: 16:9 aspect ratio (5120x2880) to later resize to 5100x2550 (17"x8.5" at 300 DPI)
            # Note: Cannot use 'size' and 'width'/'height' together per SeeDream docs
            page_num = int(page_key.split()[1]) if page_key.startswith('page ') else 0
            if page_key == 'page 0' or page_num == 12 or page_num == 13:
                # Square format for cover and coloring pages
                width = 2550
                height = 2550
            else:
                # Double-page spread format for story pages
                width = 5120
                height = 2880
            
            payload = {
                'model': self.model,
                'prompt': enhanced_prompt,
                'width': width,
                'height': height
            }
            
            # Add reference image only if provided (page 0 only in sequential mode)
            if reference_image_bytes:
                reference_image_base64 = base64.b64encode(reference_image_bytes).decode('utf-8')
                payload['reference_image'] = reference_image_base64
            
            # If there's a reference page (for sequential generation), include it as style reference
            # Prefer bytes over URL for efficiency - no need to download from our own server
            if reference_page_bytes:
                reference_page_base64 = base64.b64encode(reference_page_bytes).decode('utf-8')
                payload['style_reference_image'] = reference_page_base64
                print(f"  - Using reference page bytes directly ({len(reference_page_bytes)} bytes)")
            elif reference_page_image:
                payload['style_reference_url'] = reference_page_image
                print(f"  - Using reference page URL: {reference_page_image}")
            
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
            
            # Download and resize image to final dimensions, optionally split
            image_urls_dict = self._resize_image_to_print_size(image_url, page_key, session_id, should_split, page_number)
            
            # Return the dict containing full URL and optionally split page URLs
            return image_urls_dict
            
        except Exception as e:
            print(f"Error calling SeeDream API: {str(e)}")
            raise
    
    def _resize_image_to_print_size(self, image_url: str, page_key: str, session_id: str, should_split: bool = False, page_number: int = None) -> Dict[str, str]:
        """Download image and resize to exact physical dimensions
        Pages 0, 12, 13: 8.5" x 8.5" at 300 DPI (cover and coloring pages)
        Other pages: 17" width x 8.5" height at 300 DPI
        If should_split=True and page_key != 'page 0', split the image in the middle and save both halves as separate pages"""
        try:
            # Download the image from Seedream
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            
            # Open image with PIL
            img = Image.open(io.BytesIO(response.content))
            
            # Determine if this is a single page (cover or coloring page)
            page_num = int(page_key.split()[1]) if page_key.startswith('page ') else 0
            is_single_page = page_key == 'page 0' or page_num == 12 or page_num == 13
            
            # Physical dimensions in inches
            dpi = 300
            if is_single_page:
                # Square pages: cover and coloring pages
                width_inches = 8.5
                height_inches = 8.5
            else:
                # Double-page spread for story pages
                width_inches = 17.0
                height_inches = 8.5
            
            # Calculate pixel dimensions from physical size
            # 8.5 inches * 300 DPI = 2550 pixels (square)
            # 17 inches * 300 DPI = 5100 pixels width
            # 8.5 inches * 300 DPI = 2550 pixels height
            target_width = int(width_inches * dpi)
            target_height = int(height_inches * dpi)
            
            # Resize image to target dimensions
            # Using LANCZOS for high-quality resizing
            resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Save full image to uploads directory with proper DPI metadata
            os.makedirs('uploads/generated_images', exist_ok=True)
            page_num_str = page_key.replace('page ', '').replace(' ', '_')
            # Use session_id to make filename unique
            full_image_filename = f"uploads/generated_images/{session_id}_image_{page_num_str}.png"
            
            # Save full image with DPI information embedded
            resized_img.save(full_image_filename, format='PNG', dpi=(dpi, dpi))
            
            print(f"Saved {page_key}: {target_width}x{target_height} pixels ({width_inches}\" x {height_inches}\" at {dpi} DPI)")
            
            # Construct base URL
            base_url = os.getenv('domain') or os.getenv('BASE_URL')
            base_url = base_url.rstrip('/')
            full_image_url = f"{base_url}/{full_image_filename}"
            
            # NEVER split cover page (page 0), coloring pages (12, 13), or if splitting is disabled
            page_num = int(page_key.split()[1]) if page_key.startswith('page ') else 0
            is_single_page = page_key == 'page 0' or page_num == 12 or page_num == 13
            
            if not should_split or is_single_page:
                # For coloring pages (12, 13), return them as pages 23, 24
                # Because pages 1-11 when split = pages 1-22
                if page_num == 12:
                    return_key = 'page 23'
                elif page_num == 13:
                    return_key = 'page 24'
                else:
                    return_key = 'full'
                
                return {return_key: full_image_url}
            
            # Split the image in the middle for book pages
            # Left half: 0 to 2550 pixels (8.5" x 8.5")
            # Right half: 2550 to 5100 pixels (8.5" x 8.5")
            middle_x = target_width // 2  # 2550
            
            left_half = resized_img.crop((0, 0, middle_x, target_height))
            right_half = resized_img.crop((middle_x, 0, target_width, target_height))
            
            # Create splitted directory
            os.makedirs('uploads/generated_images/splitted', exist_ok=True)
            
            # Calculate page numbers: each image becomes 2 pages
            # page_number is the image number (0, 1, 2, 3...)
            # Image 1 -> page 1, page 2
            # Image 2 -> page 3, page 4
            left_page_num = (page_number * 2) + 1
            right_page_num = (page_number * 2) + 2
            
            # Save split images
            left_filename = f"uploads/generated_images/splitted/{session_id}_page_{left_page_num}.png"
            right_filename = f"uploads/generated_images/splitted/{session_id}_page_{right_page_num}.png"
            
            left_half.save(left_filename, format='PNG', dpi=(dpi, dpi))
            right_half.save(right_filename, format='PNG', dpi=(dpi, dpi))
            
            print(f"  - Split into page {left_page_num} and page {right_page_num} ({middle_x}x{target_height} each)")
            
            left_url = f"{base_url}/{left_filename}"
            right_url = f"{base_url}/{right_filename}"
            
            return {
                'full': full_image_url,
                f'page {left_page_num}': left_url,
                f'page {right_page_num}': right_url
            }
            
        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            # Return original URL if resize fails
            return {'full': image_url}