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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
from app.utils.image_analysis import get_text_placement_recommendation
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

class GenerateImages:
    def __init__(self):
        self.api_key = os.getenv("ARK_API_KEY")
        print(f"DEBUG: ARK_API_KEY loaded: {bool(self.api_key)} (length: {len(self.api_key) if self.api_key else 0})")
        self.model = "seedream-4-0-250828"
        self.base_url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
        self.parallel_batch_size = 5  # Generate 5 images at once in parallel mode
    
    async def _text_insertion_worker(
        self,
        text_queue: asyncio.Queue,
        results_dict: Dict,
        story: Dict[str, str],
        font_size: int,
        text_color: str,
        dpi: int,
        session_id: str,
        upload_to_s3: bool,
        book_uuid: str
    ):
        """Worker that processes text insertion queue"""
        while True:
            try:
                item = await text_queue.get()
                
                if item is None:  # Poison pill to stop worker
                    text_queue.task_done()
                    break
                
                page_key, image_bytes, page_number, is_single_page = item
                
                print(f"[Text Worker] Processing {page_key} for text insertion...")
                
                # Get story text for this page
                story_text = story.get(page_key, "")
                
                if story_text and not is_single_page:
                    # Use image analysis to add text
                    try:
                        # Get text placement recommendation
                        text_recommendation = await get_text_placement_recommendation(
                            image_bytes, 
                            story_text, 
                            font_size
                        )
                        
                        # Open image and add text
                        img = Image.open(io.BytesIO(image_bytes))
                        draw = ImageDraw.Draw(img)
                        
                        # Load font
                        from pathlib import Path
                        font_path = Path(__file__).resolve().parents[2] / "fonts" / "Comic_Relief" / "ComicRelief-Regular.ttf"
                        try:
                            if font_path.exists():
                                font = ImageFont.truetype(str(font_path), font_size)
                            else:
                                font = ImageFont.load_default()
                        except:
                            font = ImageFont.load_default()
                        
                        # Draw text with outline
                        outline_color = "black" if text_color.lower() == "white" else "white"
                        for line_coord in text_recommendation.line_coordinates:
                            x, y = line_coord.x, line_coord.y
                            line_text = line_coord.text
                            
                            # Draw outline
                            for adj_x in [-2, 0, 2]:
                                for adj_y in [-2, 0, 2]:
                                    draw.text((x + adj_x, y + adj_y), line_text, font=font, fill=outline_color)
                            # Draw main text
                            draw.text((x, y), line_text, font=font, fill=text_color)
                        
                        # Save image with text
                        output_buffer = io.BytesIO()
                        img.save(output_buffer, format='PNG', dpi=(dpi, dpi))
                        output_buffer.seek(0)
                        image_bytes = output_buffer.read()
                        
                        print(f"[Text Worker] Added text to {page_key}")
                    except Exception as e:
                        print(f"[Text Worker] Error adding text to {page_key}: {str(e)}")
                        # Continue with image without text
                
                # Now split and save the image
                img = Image.open(io.BytesIO(image_bytes))
                
                page_num_str = page_key.replace('page ', '').replace(' ', '_')
                
                # Save full image bytes
                full_image_bytes = io.BytesIO()
                img.save(full_image_bytes, format='PNG', dpi=(dpi, dpi))
                full_image_bytes.seek(0)
                full_image_bytes_data = full_image_bytes.read()
                full_image_bytes.seek(0)
                
                # Upload/save full image
                if upload_to_s3:
                    full_object_name = f"facetoon/{book_uuid}/full/image_{page_num_str}.png"
                    upload_result = upload_file_object_to_s3(full_image_bytes, object_name=full_object_name)
                    if upload_result['success']:
                        full_image_url = upload_result['url']
                    else:
                        full_image_url = None
                else:
                    os.makedirs('uploads/generated_images', exist_ok=True)
                    full_image_filename = f"uploads/generated_images/{session_id}_image_{page_num_str}.png"
                    img.save(full_image_filename, format='PNG', dpi=(dpi, dpi))
                    base_url = os.getenv('domain') or os.getenv('BASE_URL')
                    base_url = base_url.rstrip('/')
                    full_image_url = f"{base_url}/{full_image_filename}"
                
                # Split if not single page
                if not is_single_page:
                    middle_x = img.width // 2
                    left_half = img.crop((0, 0, middle_x, img.height))
                    right_half = img.crop((middle_x, 0, img.width, img.height))
                    
                    left_page_num = (page_number * 2) + 1
                    right_page_num = (page_number * 2) + 2
                    
                    if upload_to_s3:
                        # Upload split images
                        left_buffer = io.BytesIO()
                        right_buffer = io.BytesIO()
                        left_half.save(left_buffer, format='PNG', dpi=(dpi, dpi))
                        right_half.save(right_buffer, format='PNG', dpi=(dpi, dpi))
                        left_buffer.seek(0)
                        right_buffer.seek(0)
                        
                        left_bytes = left_buffer.read()
                        right_bytes = right_buffer.read()
                        left_buffer.seek(0)
                        right_buffer.seek(0)
                        
                        left_object_name = f"facetoon/{book_uuid}/splitted/page_{left_page_num}.png"
                        right_object_name = f"facetoon/{book_uuid}/splitted/page_{right_page_num}.png"
                        
                        left_result = upload_file_object_to_s3(left_buffer, object_name=left_object_name)
                        right_result = upload_file_object_to_s3(right_buffer, object_name=right_object_name)
                        
                        if left_result['success'] and right_result['success']:
                            results_dict['image_urls'][f'page {left_page_num}'] = left_result['url']
                            results_dict['image_urls'][f'page {right_page_num}'] = right_result['url']
                            results_dict['full_image_urls'][page_key] = full_image_url
                            results_dict['image_bytes'][f'page {left_page_num}'] = left_bytes
                            results_dict['image_bytes'][f'page {right_page_num}'] = right_bytes
                            print(f"[Text Worker] Completed {page_key} -> pages {left_page_num}, {right_page_num}")
                    else:
                        # Save locally
                        os.makedirs('uploads/generated_images/splitted', exist_ok=True)
                        left_filename = f"uploads/generated_images/splitted/{session_id}_page_{left_page_num}.png"
                        right_filename = f"uploads/generated_images/splitted/{session_id}_page_{right_page_num}.png"
                        
                        left_half.save(left_filename, format='PNG', dpi=(dpi, dpi))
                        right_half.save(right_filename, format='PNG', dpi=(dpi, dpi))
                        
                        left_buffer = io.BytesIO()
                        right_buffer = io.BytesIO()
                        left_half.save(left_buffer, format='PNG', dpi=(dpi, dpi))
                        right_half.save(right_buffer, format='PNG', dpi=(dpi, dpi))
                        left_buffer.seek(0)
                        right_buffer.seek(0)
                        left_bytes = left_buffer.read()
                        right_bytes = right_buffer.read()
                        
                        base_url = os.getenv('domain') or os.getenv('BASE_URL')
                        base_url = base_url.rstrip('/')
                        results_dict['image_urls'][f'page {left_page_num}'] = f"{base_url}/{left_filename}"
                        results_dict['image_urls'][f'page {right_page_num}'] = f"{base_url}/{right_filename}"
                        results_dict['full_image_urls'][page_key] = full_image_url
                        results_dict['image_bytes'][f'page {left_page_num}'] = left_bytes
                        results_dict['image_bytes'][f'page {right_page_num}'] = right_bytes
                        print(f"[Text Worker] Completed {page_key} -> pages {left_page_num}, {right_page_num}")
                else:
                    # Single page (cover or coloring)
                    page_num = int(page_key.split()[1]) if page_key.startswith('page ') else 0
                    if page_num == 12:
                        return_key = 'page 23'
                    elif page_num == 13:
                        return_key = 'page 24'
                    else:
                        return_key = page_key
                    
                    results_dict['image_urls'][return_key] = full_image_url
                    results_dict['image_bytes'][return_key] = full_image_bytes_data
                    print(f"[Text Worker] Completed {page_key} as {return_key}")
                
                text_queue.task_done()
                
            except Exception as e:
                print(f"[Text Worker] Error processing item: {str(e)}")
                import traceback
                traceback.print_exc()
                text_queue.task_done()

    async def generate_first_two_page(
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
        
        image_urls, full_image_urls, _ = await self._generate_images_for_pages(
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
    
    async def generate_images(
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
            # Skip page 0 and page 1, generate all remaining pages
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
        image_urls, full_image_urls, image_bytes = await self._generate_images_for_pages(
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
            book_uuid=book_uuid,
            font_size=100,  # Updated font size
            text_color="white",  # Default color
            dpi=300
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
    
    async def _generate_images_for_pages(
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
        book_uuid: str = None,
        font_size: int = 100,
        text_color: str = "white",
        dpi: int = 300
    ) -> tuple[Dict[str, str], Dict[str, str], Dict[str, bytes]]:
        """Generate images for specified pages using SeeDream API with async queue-based text insertion
        Returns: (image_urls, full_image_urls, image_bytes)"""
        # Create shared results dictionary for text workers
        results_dict = {
            'image_urls': {},
            'full_image_urls': {},
            'image_bytes': {}
        }
        
        # Create text insertion queue
        text_queue = asyncio.Queue()
        
        # Start text insertion workers (2 workers for parallel processing)
        num_workers = 2
        worker_tasks = []
        for i in range(num_workers):
            task = asyncio.create_task(
                self._text_insertion_worker(
                    text_queue,
                    results_dict,
                    story or {},
                    font_size,
                    text_color,
                    dpi,
                    session_id,
                    upload_to_s3,
                    book_uuid
                )
            )
            worker_tasks.append(task)
        
        print(f"[Queue System] Started {num_workers} text insertion workers")
        
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
                    # Keep reference to BytesIO to prevent garbage collection
                    img_buffer = io.BytesIO(page_1_bytes)
                    img = Image.open(img_buffer)
                    img.load()  # Force load image data before operations
                    
                    # Convert to RGB to ensure complete independence from buffer
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    middle_x = img.width // 2
                    
                    # Create independent copies of cropped regions
                    left_half = img.crop((0, 0, middle_x, img.height)).copy()
                    right_half = img.crop((middle_x, 0, img.width, img.height)).copy()
                    
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
                        
                        # Read bytes for PDF BEFORE uploading (upload closes the buffer)
                        left_bytes_for_pdf = left_buffer.read()
                        right_bytes_for_pdf = right_buffer.read()
                        left_buffer.seek(0)
                        right_buffer.seek(0)
                        
                        left_object_name = f"facetoon/{book_uuid}/splitted/page_1.png"
                        right_object_name = f"facetoon/{book_uuid}/splitted/page_2.png"
                        
                        left_result = upload_file_object_to_s3(left_buffer, object_name=left_object_name)
                        right_result = upload_file_object_to_s3(right_buffer, object_name=right_object_name)
                        
                        if left_result['success'] and right_result['success']:
                            image_urls['page 1'] = left_result['url']
                            image_urls['page 2'] = right_result['url']
                            # Store bytes for PDF (already read before upload)
                            image_bytes_for_pdf['page 1'] = left_bytes_for_pdf
                            image_bytes_for_pdf['page 2'] = right_bytes_for_pdf
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
                import traceback
                print(f"Error loading/splitting page 1 image: {str(e)}")
                traceback.print_exc()
        
        page_counter = page_counter_start
        
        if force_sequential:
            # Sequential generation when forced
            sorted_pages = sorted(pages.items(), key=lambda x: int(x[0].split()[1]))
            
            print(f"\nSequential generation: Processing {len(sorted_pages)} pages from prompts")
            print(f"Pages to generate: {[k for k, _ in sorted_pages]}")
            print(f"Starting page_counter: {page_counter}")
            
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
                print(f"  - Reference page URL: {reference_page if reference_page else 'None'}")
                print(f"  - Page number for splitting: {current_page_number}")
                
                # Generate image and get resized bytes
                image_bytes, is_single_page = await asyncio.to_thread(
                    self._generate_single_image,
                    prompt,
                    reference_images_bytes_list if use_raw_image else None,
                    reference_page,
                    gender,
                    age,
                    image_style,
                    page_key,
                    session_id,
                    reference_page_bytes=reference_page_bytes,
                    page_number=current_page_number
                )
                
                # Add to text insertion queue immediately
                await text_queue.put((page_key, image_bytes, current_page_number, is_single_page))
                print(f"[Queue System] Added {page_key} to text insertion queue")
                
                # Store bytes for sequential reference (next page needs previous page)
                generated_images_bytes[page_key] = image_bytes
                
                # Increment page counter for non-cover pages
                if page_key != 'page 0' and not is_single_page:
                    page_counter += 1
            
            # Clear generated_images dict to free memory after sequential generation
            generated_images.clear()
            print("Cleared generated_images cache after sequential generation")
        else:
            # Parallel generation - all pages use the reference image
            # Generate all images concurrently
            print(f"Parallel generation mode: Generating {len(pages)} pages concurrently")
            print(f"All pages will use the reference image, ignoring any page connections")
            
            async def generate_and_queue(page_key, prompt, page_num_for_split):
                """Generate image and add to text queue"""
                try:
                    image_bytes, is_single_page = await asyncio.to_thread(
                        self._generate_single_image,
                        prompt,
                        reference_images_bytes_list,  # All pages get reference images in parallel mode
                        None,  # No previous page reference in parallel mode
                        gender,
                        age,
                        image_style,
                        page_key,
                        session_id,
                        page_number=page_num_for_split
                    )
                    
                    # Add to text insertion queue
                    await text_queue.put((page_key, image_bytes, page_num_for_split, is_single_page))
                    print(f"[Queue System] Added {page_key} to text insertion queue")
                    
                except Exception as e:
                    print(f"Error generating image for {page_key}: {str(e)}")
                    raise Exception(f"Failed to generate image for {page_key}: {str(e)}")
            
            # Create tasks for all pages
            generation_tasks = []
            current_page_counter = page_counter
            for page_key, prompt in pages.items():
                # Determine page_number for splitting calculation
                if page_key == 'page 0':
                    page_num_for_split = 0
                else:
                    page_num_for_split = current_page_counter
                    current_page_counter += 1
                
                task = generate_and_queue(page_key, prompt, page_num_for_split)
                generation_tasks.append(task)
            
            # Wait for all generations to complete
            await asyncio.gather(*generation_tasks)
        
        # Signal workers to stop by adding poison pills
        print(f"\n[Queue System] All images generated, waiting for text insertion to complete...")
        for _ in range(num_workers):
            await text_queue.put(None)
        
        # Wait for queue to be empty and workers to finish
        await text_queue.join()
        await asyncio.gather(*worker_tasks)
        
        print(f"[Queue System] All text insertions completed")
        
        # Extract results from shared dictionary
        image_urls = results_dict['image_urls']
        full_image_urls = results_dict['full_image_urls']
        image_bytes_for_pdf = results_dict['image_bytes']
        
        print(f"\nFinal image_urls dictionary contains {len(image_urls)} pages:")
        for key in sorted(image_urls.keys(), key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else 0):
            if not key.endswith('_bytes'):
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
        session_id: str = None,
        reference_page_bytes: Optional[bytes] = None,
        page_number: int = None
    ) -> tuple[bytes, bool]:
        """Generate a single image using SeeDream API and return resized bytes
        Returns: (image_bytes, is_single_page)"""
        try:
            # Enhance the prompt with detailed style and character instructions
            if page_key == "page 0":
                # Cover page - no text generation, composition only
                enhanced_prompt = f"""
Children's storybook cover illustration in {image_style} style.
Main character: {age}-year-old {gender} child matching the reference image exactly.
{prompt}
Composition suitable for a book cover with space for title text placement.
Style: Professional children's book illustration, vibrant colors, high quality, storybook art, child-friendly, whimsical and engaging.
The child's face, features, hair, and appearance must exactly match the reference image provided.
DO NOT include any text or letters in the image.
""".strip()
            else:
                # Check if this is a coloring page (pages 12 or 13)
                page_num = int(page_key.split()[1]) if page_key.startswith('page ') else 0
                is_coloring_page = page_num == 12 or page_num == 13
                
                # Story pages - no text generation, image only
                enhanced_prompt = f"""
Children's storybook illustration in {image_style} style.
Main character: {age}-year-old {gender} child continuing from the previous page.
{prompt}
Composition suitable for a children's storybook with space for text placement on the left side.
Style: Professional children's book illustration, vibrant colors, high quality, storybook art, child-friendly, whimsical and engaging.
CRITICAL: Maintain EXACT character appearance from the style reference - same facial features, eyebrows, eye shape, nose, mouth, hair color, hair style, and skin tone. If clothing colors or patterns are specified in the prompt, follow them precisely without variation.
Focus on: implementing the exact scene, actions, setting, and clothing details as described while preserving all character appearance characteristics.
DO NOT include any text, letters, or words in the image.
Negative prompt: No changes to the character's face structure, facial proportions, eyebrow thickness or shape, eye color or shape, nose shape, mouth shape, hair color, hair style, or skin tone. Do not modify clothing colors from the prompt description. No artistic reinterpretation of the character's established appearance. No text or letters in the image.
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
            
            # Debug: Print response details before raising error
            if response.status_code != 200:
                print(f"API Error Response:")
                print(f"  Status Code: {response.status_code}")
                print(f"  Response Body: {response.text}")
                try:
                    error_json = response.json()
                    print(f"  Response JSON: {error_json}")
                except:
                    pass
            
            response.raise_for_status()
            result = response.json()
            
            # Extract image URL from response (adjust based on actual API response format)
            image_url = result.get('data', [{}])[0].get('url') or result.get('image_url') or result.get('url')
            
            if not image_url:
                raise Exception(f"No image URL in response: {result}")
            
            # Download and resize image to final dimensions
            image_bytes, is_single_page = self._resize_image_to_print_size(image_url, page_key, session_id)
            
            # Return bytes and is_single_page flag
            return image_bytes, is_single_page
            
        except Exception as e:
            print(f"Error calling SeeDream API: {str(e)}")
            raise
    
    def _resize_image_to_print_size(self, image_url: str, page_key: str, session_id: str) -> tuple[bytes, bool]:
        """Download image and resize to exact physical dimensions
        Pages 0, 12, 13: 8.5" x 8.5" at 300 DPI (cover and coloring pages)
        Other pages: 17" width x 8.5" height at 300 DPI
        Returns: (image_bytes, is_single_page)"""
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
            page_num = int(page_key.split()[1]) if page_key.startswith('page ') else 0
            is_single_page = page_key == 'page 0' or page_num == 12 or page_num == 13
            
            if is_single_page:
                # Square pages: cover and coloring pages
                width_inches = 8.5
                height_inches = 8.5
            else:
                # Double-page spread for story pages
                width_inches = 17.0
                height_inches = 8.5
            
            # Calculate pixel dimensions from physical size
            target_width = int(width_inches * dpi)
            target_height = int(height_inches * dpi)
            
            # Resize image to target dimensions using LANCZOS for high-quality resizing
            resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Convert to bytes
            output_buffer = io.BytesIO()
            resized_img.save(output_buffer, format='PNG', dpi=(dpi, dpi))
            output_buffer.seek(0)
            image_bytes = output_buffer.read()
            
            print(f"Generated {page_key}: {target_width}x{target_height} pixels ({width_inches}\" x {height_inches}\" at {dpi} DPI)")
            
            # Return bytes and whether this is a single page
            return image_bytes, is_single_page
            
        except Exception as e:
            print(f"Error generating/resizing image: {str(e)}")
            raise Exception(f"Failed to generate image: {str(e)}")
    
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
                        
                        # Add second logo page (Logo_2.png)
                        try:
                            logo_2_path = 'Logo_2.png'
                            if os.path.exists(logo_2_path):
                                logo_2_img = Image.open(logo_2_path)
                                
                                # Convert to RGB if needed
                                if logo_2_img.mode != 'RGB':
                                    logo_2_img = logo_2_img.convert('RGB')
                                
                                # Resize to fit page
                                logo_2_img = logo_2_img.resize((612, 612), Image.Resampling.LANCZOS)
                                
                                # Convert to ImageReader
                                logo_2_buffer = io.BytesIO()
                                logo_2_img.save(logo_2_buffer, format='PNG')
                                logo_2_buffer.seek(0)
                                logo_2_reader = ImageReader(logo_2_buffer)
                                
                                # Draw second logo page
                                c.drawImage(logo_2_reader, 0, 0, width=612, height=612)
                                c.showPage()
                                
                                print(f"  - Added Logo_2 page to PDF")
                            else:
                                print(f"  - Warning: Logo_2.png not found in root folder")
                        except Exception as e:
                            print(f"  - Error adding logo_2 page: {str(e)}")
                    
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