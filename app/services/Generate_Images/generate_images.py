import os
import json
import requests
from dotenv import load_dotenv
from .generate_images_schema import GenerateImageResponse, PageImageUrls
from typing import Dict, Optional, List
from fastapi import UploadFile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io
import base64
from urllib.parse import urljoin
import uuid
from datetime import datetime
from app.utils.upload_to_bucket import upload_file_to_s3, upload_file_object_to_s3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

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
        reference_images: List[UploadFile],
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
        
        image_urls, full_image_urls, _ = self._generate_images_for_pages(
            pages_to_generate,
            reference_images,
            page_connections=page_connections if force_sequential else None,
            gender=gender,
            age=age,
            image_style=image_style,
            force_sequential=force_sequential,
            story=story_to_generate,
            session_id=session_id,
            should_split=True  # Split page 1 into page 1 and page 2
        )
        
        # Convert dict to structured format
        structured_urls = self._convert_dict_to_structured(image_urls, full_image_urls)
        return GenerateImageResponse(image_urls=structured_urls)
    
    def generate_images(
        self,
        prompts: Dict[str, str],
        page_connections: Optional[Dict[str, str]],
        reference_images: Optional[List[UploadFile]],
        gender: str,
        age: int,
        image_style: str,
        coverpage: str = "no",
        sequential: str = "no",
        story: Optional[Dict[str, str]] = None,
        page_0_url: Optional[str] = None,
        page_1_url: Optional[str] = None
    ) -> GenerateImageResponse:
        """Generate images for all pages or skip cover/page 1 if they exist"""
        # Generate unique session ID for this request
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        # Generate unique book UUID for S3 directory structure
        book_uuid = str(uuid.uuid4())
        
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
        image_urls, full_image_urls, image_bytes = self._generate_images_for_pages(
            pages_to_generate,
            reference_images,
            page_connections if force_sequential else None,
            gender=gender,
            age=age,
            image_style=image_style,
            force_sequential=force_sequential,
            story=story_to_generate,
            session_id=session_id,
            page_0_url=page_0_url,
            page_1_url=page_1_url,
            should_split=True,  # Enable splitting for full generation
            upload_to_s3=True,  # Upload to S3 for full generation
            book_uuid=book_uuid
        )
        
        # Convert dict to structured format
        structured_urls = self._convert_dict_to_structured(image_urls, full_image_urls)
        
        # Generate PDF with all pages
        pdf_url = self._generate_pdf(
            image_bytes=image_bytes,
            book_uuid=book_uuid,
            session_id=session_id,
            upload_to_s3=True
        )
        
        return GenerateImageResponse(image_urls=structured_urls, pdf_url=pdf_url)
    
    def _generate_images_for_pages(
        self,
        pages: Dict[str, str],
        reference_images: Optional[List[UploadFile]],
        page_connections: Optional[Dict[str, str]],
        gender: str,
        age: int,
        image_style: str,
        force_sequential: bool = False,
        story: Optional[Dict[str, str]] = None,
        session_id: str = None,
        page_0_url: Optional[str] = None,
        page_1_url: Optional[str] = None,
        should_split: bool = False,
        upload_to_s3: bool = False,
        book_uuid: str = None
    ) -> tuple[Dict[str, str], Dict[str, str], Dict[str, bytes]]:
        """Generate images for specified pages using SeeDream API
        Returns: (image_urls, full_image_urls, image_bytes)"""
        image_urls = {}
        full_image_urls = {}  # Store full image URLs separately
        image_bytes_for_pdf = {}  # Store image bytes for PDF generation
        generated_images = {}  # Initialize for sequential generation
        generated_images_bytes = {}  # Store bytes for direct use
        story = story or {}
        
        # Determine starting page counter based on whether page_1_url is provided
        # If page_1_url is provided and will be split into pages 1-2, next generated image starts at image 1 (pages 3-4)
        # If no page_1_url, first generated image is image 0 for page 0, or image 1 for pages 1-2 (if skipping cover)
        page_counter_start = 0
        
        # Read all reference images
        reference_images_bytes_list = []
        if reference_images:
            for ref_img in reference_images:
                img_bytes = ref_img.file.read()
                ref_img.file.seek(0)  # Reset file pointer
                reference_images_bytes_list.append(img_bytes)
        
        # Handle page_0_url if provided - upload to S3
        if page_0_url and upload_to_s3:
            try:
                # Extract filename from URL and read from local uploads folder
                if page_0_url.startswith('http'):
                    from urllib.parse import urlparse
                    parsed = urlparse(page_0_url)
                    file_path = parsed.path.lstrip('/')
                else:
                    file_path = page_0_url.lstrip('/')
                
                # Read the image file directly from local disk
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        page_0_bytes = f.read()
                else:
                    print(f"Warning: Page 0 image not found at: {file_path}")
                    page_0_bytes = None
                
                if page_0_bytes:
                    # Upload to S3
                    page_0_object_name = f"facetoon/{book_uuid}/page_0.png"
                    page_0_buffer = io.BytesIO(page_0_bytes)
                    upload_result = upload_file_object_to_s3(page_0_buffer, object_name=page_0_object_name)
                    
                    if upload_result['success']:
                        image_urls['page 0'] = upload_result['url']
                        image_bytes_for_pdf['page 0'] = page_0_bytes
                        print(f"Uploaded page 0 to S3: {upload_result['url']}")
                    else:
                        print(f"Failed to upload page 0 to S3: {upload_result['message']}")
            except Exception as e:
                print(f"Error processing page 0 URL: {str(e)}")
        
        # If page_1_url is provided, read the image from local disk and split it
        if page_1_url and should_split:
            try:
                # Extract filename from URL and read from local uploads folder
                if page_1_url.startswith('http'):
                    from urllib.parse import urlparse
                    parsed = urlparse(page_1_url)
                    file_path = parsed.path.lstrip('/')
                else:
                    file_path = page_1_url.lstrip('/')
                
                # Read the image file directly from local disk
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
                    
                    if upload_to_s3:
                        # Upload full page 1 image to S3 first
                        full_page_1_buffer = io.BytesIO(page_1_bytes)
                        full_page_1_object_name = f"facetoon/{book_uuid}/full/image_1.png"
                        full_page_1_result = upload_file_object_to_s3(full_page_1_buffer, object_name=full_page_1_object_name)
                        
                        if full_page_1_result['success']:
                            full_image_urls['page 1'] = full_page_1_result['url']
                            print(f"Uploaded full page 1 to S3: {full_page_1_result['url']}")
                        
                        # Upload split images to S3
                        left_buffer = io.BytesIO()
                        right_buffer = io.BytesIO()
                        left_half.save(left_buffer, format='PNG', dpi=(300, 300))
                        right_half.save(right_buffer, format='PNG', dpi=(300, 300))
                        left_buffer.seek(0)
                        right_buffer.seek(0)
                        
                        left_object_name = f"facetoon/{book_uuid}/splitted/page_1.png"
                        right_object_name = f"facetoon/{book_uuid}/splitted/page_2.png"
                        
                        left_result = upload_file_object_to_s3(left_buffer, object_name=left_object_name)
                        right_result = upload_file_object_to_s3(right_buffer, object_name=right_object_name)
                        
                        if left_result['success'] and right_result['success']:
                            image_urls['page 1'] = left_result['url']
                            image_urls['page 2'] = right_result['url']
                            # Store bytes for PDF
                            left_buffer.seek(0)
                            right_buffer.seek(0)
                            image_bytes_for_pdf['page 1'] = left_buffer.read()
                            image_bytes_for_pdf['page 2'] = right_buffer.read()
                            print(f"Uploaded split page 1 to S3")
                            print(f"  - page 1: {left_result['url']}")
                            print(f"  - page 2: {right_result['url']}")
                        else:
                            print(f"Failed to upload split pages to S3")
                    else:
                        # Save locally
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
                print(f"  - Has reference bytes: {bool(reference_page_bytes)}")
                print(f"  - Reference page URL: {reference_page if reference_page else 'None'}")
                print(f"  - Page number for splitting: {current_page_number}")
                
                image_urls_dict = self._generate_single_image(
                    prompt,
                    reference_images_bytes_list if use_raw_image else None,
                    reference_page,
                    gender,
                    age,
                    image_style,
                    page_key,
                    story.get(page_key),
                    session_id,
                    reference_page_bytes=reference_page_bytes,
                    should_split=should_split,
                    page_number=current_page_number,
                    upload_to_s3=upload_to_s3,
                    book_uuid=book_uuid
                )
                
                # Store full URL and bytes for AI reference in next pages
                # For pages with custom keys (coloring pages), get the actual URL value
                if 'full' in image_urls_dict:
                    generated_images[page_key] = image_urls_dict['full']
                    print(f"  - Generated full URL: {image_urls_dict['full']}")
                else:
                    # Get the first URL value (for coloring pages with custom keys)
                    url_value = next(iter(image_urls_dict.values()))
                    generated_images[page_key] = url_value
                    print(f"  - Generated URL: {url_value}")
                
                # Store the generated image bytes for next page reference
                # Download the full image and store bytes for sequential generation
                if 'full_bytes' in image_urls_dict:
                    generated_images_bytes[page_key] = image_urls_dict['full_bytes']
                
                # Store bytes for PDF generation
                if should_split and page_key != 'page 0':
                    # For split pages, store the left and right halves
                    page_num = int(page_key.split()[1]) if page_key.startswith('page ') else 0
                    is_coloring_page = page_num == 12 or page_num == 13
                    
                    if not is_coloring_page:
                        # Extract split page bytes
                        for key in image_urls_dict.keys():
                            if key.startswith('page ') and key != 'full' and key != 'full_bytes':
                                if f'{key}_bytes' in image_urls_dict:
                                    image_bytes_for_pdf[key] = image_urls_dict[f'{key}_bytes']
                    else:
                        # Coloring pages - store with their actual key (page 23, 24)
                        if 'full_bytes' in image_urls_dict:
                            for key in image_urls_dict.keys():
                                if key.startswith('page ') and key != 'full':
                                    image_bytes_for_pdf[key] = image_urls_dict['full_bytes']
                else:
                    # Cover page or non-split page
                    if 'full_bytes' in image_urls_dict:
                        image_bytes_for_pdf[page_key] = image_urls_dict['full_bytes']
                
                # Add URLs to response based on whether page is split
                if should_split and page_key != 'page 0':
                    # Check if page was actually split or returned with a different key
                    page_num = int(page_key.split()[1]) if page_key.startswith('page ') else 0
                    is_coloring_page = page_num == 12 or page_num == 13
                    
                    if is_coloring_page:
                        # Coloring pages are not split, add with their custom key (page 23, 24)
                        for key, url in image_urls_dict.items():
                            if key != 'full' and key != 'full_bytes':
                                image_urls[key] = url
                                print(f"  - {key}: {url}")
                    else:
                        # Split pages - add individual page URLs and store full URL
                        if 'full' in image_urls_dict:
                            full_image_urls[page_key] = image_urls_dict['full']
                        for key, url in image_urls_dict.items():
                            if key != 'full' and key != 'full_bytes':
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
                        reference_images_bytes_list,  # All pages get reference images in parallel mode
                        None,  # No previous page reference in parallel mode
                        gender,
                        age,
                        image_style,
                        page_key,
                        story.get(page_key),
                        session_id,
                        should_split=should_split,
                        page_number=page_num_for_split,
                        upload_to_s3=upload_to_s3,
                        book_uuid=book_uuid
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
                                if key != 'full' and key != 'full_bytes':
                                    image_urls[key] = url
                                    print(f"  - {key}: {url}")
                        else:
                            # Page was not split (page 0, coloring pages, or splitting disabled)
                            # Check if dict has a specific key (like 'page 23' for coloring pages)
                            for key, url in image_urls_dict.items():
                                if key != 'full' and key != 'full_bytes':
                                    # Use the specific key (e.g., 'page 23', 'page 24')
                                    image_urls[key] = url
                                    print(f"  - {key}: {url}")
                                elif key == 'full':
                                    # No specific key, use original page_key
                                    image_urls[page_key] = url
                                    print(f"  - {page_key}: {url}")
                    except Exception as e:
                        print(f"Error generating image for {page_key}: {str(e)}")
                        raise Exception(f"Failed to generate image for {page_key}: {str(e)}")
        
        print(f"\nFinal image_urls dictionary contains {len(image_urls)} pages:")
        for key in sorted(image_urls.keys(), key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else 0):
            print(f"  - {key}")
        
        return image_urls, full_image_urls, image_bytes_for_pdf
    
    def _generate_single_image(
        self,
        prompt: str,
        reference_images_bytes: Optional[List[bytes]],
        reference_page_image: Optional[str],
        gender: str,
        age: int,
        image_style: str,
        page_key: str,
        story_text: Optional[str] = None,
        session_id: str = None,
        reference_page_bytes: Optional[bytes] = None,
        should_split: bool = False,
        page_number: int = None,
        upload_to_s3: bool = False,
        book_uuid: str = None
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
                    text_instruction = f"""
CRITICAL TEXT RENDERING REQUIREMENT:
You MUST include the COMPLETE text exactly as written below in the generated image. Do not truncate, shorten, or cut off any words.
The ENTIRE text MUST be visible and readable in the image.

FULL TEXT TO RENDER (COMPLETE, NO TRUNCATION):
"{story_text}"

TEXT PLACEMENT INSTRUCTIONS - DOUBLE PAGE SPREAD:
IMPORTANT: This image is for a double-page spread (two pages side by side). The image is 17" wide (two 8.5" pages).
- Place the text ONLY on the LEFT HALF of the image (the left 8.5" section)
- The text should be contained within the LEFT PAGE only (left 50% of the image width)
- Do NOT place text on the right half of the image
- Position the text in a clear, readable area on the left page (top, middle, or bottom)
- Use a clear, legible font size that fits the entire text within the left page area
- Ensure ALL words from the beginning to the end are fully visible on the left page
- The text must be complete from start to finish: "{story_text}"
- If needed, use multiple lines to fit all the text within the left page, but ALL text must be included
- Do NOT abbreviate, truncate, or use ellipsis (...) - render the FULL text
- Keep the right half of the image (right page) for the illustration only, without any text
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
                'height': height,
                'watermark': False
            }
            
            # Add reference images if provided (up to 3)
            if reference_images_bytes and len(reference_images_bytes) > 0:
                for idx, img_bytes in enumerate(reference_images_bytes[:3], 1):
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    if idx == 1:
                        payload['reference_image'] = img_base64
                    else:
                        payload[f'reference_image_{idx}'] = img_base64
            
            # If there's a reference page (for sequential generation), include it as style reference
            # Prefer bytes over URL for efficiency - no need to download from our own server
            if reference_page_bytes:
                reference_page_base64 = base64.b64encode(reference_page_bytes).decode('utf-8')
                payload['style_reference_image'] = reference_page_base64
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
            image_urls_dict = self._resize_image_to_print_size(image_url, page_key, session_id, should_split, page_number, upload_to_s3, book_uuid)
            
            # Return the dict containing full URL and optionally split page URLs
            return image_urls_dict
            
        except Exception as e:
            print(f"Error calling SeeDream API: {str(e)}")
            raise
    
    def _resize_image_to_print_size(self, image_url: str, page_key: str, session_id: str, should_split: bool = False, page_number: int = None, upload_to_s3: bool = False, book_uuid: str = None) -> Dict[str, str]:
        """Download image and resize to exact physical dimensions
        Pages 0, 12, 13: 8.5" x 8.5" at 300 DPI (cover and coloring pages)
        Other pages: 17" width x 8.5" height at 300 DPI
        If should_split=True and page_key != 'page 0', split the image in the middle and save both halves as separate pages
        If upload_to_s3=True, upload images to S3 in facetoon/{book_uuid}/ directory"""
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
            
            page_num_str = page_key.replace('page ', '').replace(' ', '_')
            
            # Store full image bytes for sequential generation reference
            full_image_bytes = io.BytesIO()
            resized_img.save(full_image_bytes, format='PNG', dpi=(dpi, dpi))
            full_image_bytes.seek(0)
            full_image_bytes_data = full_image_bytes.read()
            full_image_bytes.seek(0)  # Reset for upload
            
            if upload_to_s3:
                # Upload full image to S3
                full_object_name = f"facetoon/{book_uuid}/full/image_{page_num_str}.png"
                upload_result = upload_file_object_to_s3(full_image_bytes, object_name=full_object_name)
                
                if upload_result['success']:
                    full_image_url = upload_result['url']
                    print(f"Uploaded {page_key} to S3: {target_width}x{target_height} pixels ({width_inches}\" x {height_inches}\" at {dpi} DPI)")
                else:
                    print(f"Failed to upload {page_key} to S3: {upload_result['message']}")
                    # Fallback to original URL
                    full_image_url = image_url
            else:
                # Save full image to uploads directory with proper DPI metadata
                os.makedirs('uploads/generated_images', exist_ok=True)
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
                
                return {return_key: full_image_url, 'full_bytes': full_image_bytes_data}
            
            # Split the image in the middle for book pages
            # Left half: 0 to 2550 pixels (8.5" x 8.5")
            # Right half: 2550 to 5100 pixels (8.5" x 8.5")
            middle_x = target_width // 2  # 2550
            
            left_half = resized_img.crop((0, 0, middle_x, target_height))
            right_half = resized_img.crop((middle_x, 0, target_width, target_height))
            
            # Calculate page numbers: each image becomes 2 pages
            # page_number is the image number (0, 1, 2, 3...)
            # Image 1 -> page 1, page 2
            # Image 2 -> page 3, page 4
            left_page_num = (page_number * 2) + 1
            right_page_num = (page_number * 2) + 2
            
            if upload_to_s3:
                # Upload split images to S3
                left_buffer = io.BytesIO()
                right_buffer = io.BytesIO()
                left_half.save(left_buffer, format='PNG', dpi=(dpi, dpi))
                right_half.save(right_buffer, format='PNG', dpi=(dpi, dpi))
                left_buffer.seek(0)
                right_buffer.seek(0)
                
                # Read bytes for PDF before uploading
                left_bytes = left_buffer.read()
                right_bytes = right_buffer.read()
                left_buffer.seek(0)  # Reset for upload
                right_buffer.seek(0)
                
                left_object_name = f"facetoon/{book_uuid}/splitted/page_{left_page_num}.png"
                right_object_name = f"facetoon/{book_uuid}/splitted/page_{right_page_num}.png"
                
                left_result = upload_file_object_to_s3(left_buffer, object_name=left_object_name)
                right_result = upload_file_object_to_s3(right_buffer, object_name=right_object_name)
                
                if left_result['success'] and right_result['success']:
                    left_url = left_result['url']
                    right_url = right_result['url']
                    print(f"  - Split and uploaded to S3: page {left_page_num} and page {right_page_num} ({middle_x}x{target_height} each)")
                else:
                    print(f"  - Failed to upload split images to S3")
                    # Fallback to full image URL
                    left_url = full_image_url
                    right_url = full_image_url
            else:
                # Create splitted directory
                os.makedirs('uploads/generated_images/splitted', exist_ok=True)
                
                # Save split images
                left_filename = f"uploads/generated_images/splitted/{session_id}_page_{left_page_num}.png"
                right_filename = f"uploads/generated_images/splitted/{session_id}_page_{right_page_num}.png"
                
                left_half.save(left_filename, format='PNG', dpi=(dpi, dpi))
                right_half.save(right_filename, format='PNG', dpi=(dpi, dpi))
                
                # Read bytes for PDF
                left_buffer = io.BytesIO()
                right_buffer = io.BytesIO()
                left_half.save(left_buffer, format='PNG', dpi=(dpi, dpi))
                right_half.save(right_buffer, format='PNG', dpi=(dpi, dpi))
                left_buffer.seek(0)
                right_buffer.seek(0)
                left_bytes = left_buffer.read()
                right_bytes = right_buffer.read()
                
                print(f"  - Split into page {left_page_num} and page {right_page_num} ({middle_x}x{target_height} each)")
                
                base_url = os.getenv('domain') or os.getenv('BASE_URL')
                base_url = base_url.rstrip('/')
                left_url = f"{base_url}/{left_filename}"
                right_url = f"{base_url}/{right_filename}"
            
            return {
                'full': full_image_url,
                'full_bytes': full_image_bytes_data,
                f'page {left_page_num}': left_url,
                f'page {right_page_num}': right_url,
                f'page {left_page_num}_bytes': left_bytes,
                f'page {right_page_num}_bytes': right_bytes
            }
            
        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            # Return original URL if resize fails
            return {'full': image_url}
    
    def _convert_dict_to_structured(self, image_urls: Dict[str, str], full_image_urls: Dict[str, str]) -> List[PageImageUrls]:
        """Convert flat dictionary to structured format with PageImageUrls objects"""
        structured_pages = []
        processed_pages = set()
        
        # Sort keys by page number
        sorted_keys = sorted(image_urls.keys(), key=lambda x: int(x.split()[1]) if 'page' in x and x.split()[1].isdigit() else 0)
        
        for key in sorted_keys:
            url = image_urls[key]
            
            if 'page' in key:
                page_num_str = key.split()[1]
                if page_num_str.isdigit():
                    page_num = int(page_num_str)
                    
                    # Skip if already processed
                    if page_num in processed_pages:
                        continue
                    
                    # Check if this is part of a split pair (odd pages 1, 3, 5, etc. pair with even pages 2, 4, 6, etc.)
                    if page_num % 2 == 1 and page_num > 0:
                        right_page_num = page_num + 1
                        right_key = f'page {right_page_num}'
                        
                        if right_key in image_urls:
                            # This is a split page pair
                            # Reconstruct which original page this came from
                            # pages 1-2 come from page 1, pages 3-4 from page 2, etc.
                            original_page_num = (page_num + 1) // 2
                            original_page_key = f'page {original_page_num}'
                            
                            # Get full URL from full_image_urls dict
                            full_url = full_image_urls.get(original_page_key, "")
                            
                            page_obj = PageImageUrls(
                                name=original_page_key,
                                fullPageUrl=full_url,
                                leftUrl=url,
                                rightUrl=image_urls[right_key]
                            )
                            structured_pages.append(page_obj)
                            processed_pages.add(page_num)
                            processed_pages.add(right_page_num)
                        else:
                            # Single page
                            page_obj = PageImageUrls(
                                name=key,
                                fullPageUrl=url,
                                leftUrl=None,
                                rightUrl=None
                            )
                            structured_pages.append(page_obj)
                            processed_pages.add(page_num)
                    elif page_num == 0 or page_num >= 23 or page_num % 2 == 0:
                        # Cover page, coloring pages, or even numbered pages (if not already paired)
                        if page_num not in processed_pages:
                            page_obj = PageImageUrls(
                                name=key,
                                fullPageUrl=url,
                                leftUrl=None,
                                rightUrl=None
                            )
                            structured_pages.append(page_obj)
                            processed_pages.add(page_num)
        
        return structured_pages
    
    def _generate_pdf(
        self,
        image_bytes: Dict[str, bytes],
        book_uuid: str,
        session_id: str,
        upload_to_s3: bool = False
    ) -> Optional[str]:
        """Generate a PDF book with cover, logo, and all pages
        Each page is 8.5\" x 8.5\" at 300 DPI"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas as pdf_canvas
            from reportlab.lib.utils import ImageReader
            
            # Page size: 8.5" x 8.5" at 72 points per inch
            page_size = (8.5 * 72, 8.5 * 72)  # 612 x 612 points
            
            # Create PDF in memory
            pdf_buffer = io.BytesIO()
            c = pdf_canvas.Canvas(pdf_buffer, pagesize=page_size)
            
            # Sort pages by number
            sorted_pages = sorted(
                [(k, v) for k, v in image_bytes.items() if k.startswith('page ')],
                key=lambda x: int(x[0].split()[1])
            )
            
            print(f"\\nGenerating PDF with {len(sorted_pages)} pages + logo page...")
            
            # Add each page to PDF
            for page_key, page_bytes in sorted_pages:
                try:
                    # Open image from bytes
                    img = Image.open(io.BytesIO(page_bytes))
                    
                    # Convert to RGB if needed (PDF doesn't support RGBA)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to 8.5\" x 8.5\" at 72 DPI for PDF
                    img = img.resize((612, 612), Image.Resampling.LANCZOS)
                    
                    # Convert to ImageReader for reportlab
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    img_reader = ImageReader(img_buffer)
                    
                    # Draw image on PDF page
                    c.drawImage(img_reader, 0, 0, width=612, height=612)
                    c.showPage()
                    
                    print(f"  - Added {page_key} to PDF")
                    
                    # Add logo page after cover (page 0)
                    if page_key == 'page 0':
                        try:
                            logo_path = 'Logo.png'
                            if os.path.exists(logo_path):
                                logo_img = Image.open(logo_path)
                                
                                # Convert to RGB if needed
                                if logo_img.mode != 'RGB':
                                    logo_img = logo_img.convert('RGB')
                                
                                # Resize to fit page
                                logo_img = logo_img.resize((612, 612), Image.Resampling.LANCZOS)
                                
                                # Convert to ImageReader
                                logo_buffer = io.BytesIO()
                                logo_img.save(logo_buffer, format='PNG')
                                logo_buffer.seek(0)
                                logo_reader = ImageReader(logo_buffer)
                                
                                # Draw logo page
                                c.drawImage(logo_reader, 0, 0, width=612, height=612)
                                c.showPage()
                                
                                print(f"  - Added Logo page to PDF")
                            else:
                                print(f"  - Warning: Logo.png not found in root folder")
                        except Exception as e:
                            print(f"  - Error adding logo page: {str(e)}")
                    
                except Exception as e:
                    print(f"Error adding {page_key} to PDF: {str(e)}")
                    continue
            
            # Save PDF
            c.save()
            pdf_buffer.seek(0)
            
            # Upload to S3 or save locally
            if upload_to_s3:
                pdf_object_name = f"facetoon/{book_uuid}/book_{session_id}.pdf"
                upload_result = upload_file_object_to_s3(pdf_buffer, object_name=pdf_object_name)
                
                if upload_result['success']:
                    print(f"\nPDF uploaded to S3: {upload_result['url']}")
                    return upload_result['url']
                else:
                    print(f"Failed to upload PDF to S3: {upload_result['message']}")
                    return None
            else:
                # Save locally
                os.makedirs('uploads/generated_pdfs', exist_ok=True)
                pdf_filename = f"uploads/generated_pdfs/{session_id}_book.pdf"
                
                with open(pdf_filename, 'wb') as f:
                    f.write(pdf_buffer.read())
                
                base_url = os.getenv('domain') or os.getenv('BASE_URL')
                base_url = base_url.rstrip('/')
                pdf_url = f"{base_url}/{pdf_filename}"
                
                print(f"\nPDF saved locally: {pdf_url}")
                return pdf_url
                
        except Exception as e:
            print(f"Error generating PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return None