import os
import json
import requests
import time
import traceback
from dotenv import load_dotenv
from .generate_images_schema import GenerateImageResponse, PageImageUrls
from typing import Dict, Optional, List
from fastapi import UploadFile
import asyncio
from PIL import Image
import io
import base64
import uuid
from datetime import datetime
from app.utils.upload_to_bucket import upload_file_to_s3, upload_file_object_to_s3
from app.utils.image_processing import resize_image_to_print_size, convert_dict_to_structured
from app.utils.pdf_generator import generate_pdf

load_dotenv()

class GenerateImages:
    def __init__(self):
        self.api_key = os.getenv("ARK_API_KEY")
        print(f"DEBUG: ARK_API_KEY loaded: {bool(self.api_key)} (length: {len(self.api_key) if self.api_key else 0})")
        self.model = "seedream-4-0-250828"
        self.base_url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
        self.parallel_batch_size = 5  # Generate 5 images at once in parallel mode
    
    @staticmethod
    def _get_page_number(page_key: str) -> int:
        """Extract page number from page_key, mapping 'page last page' to 14"""
        if page_key == 'page last page':
            return 14  # Back cover page
        elif page_key.startswith('page ') and page_key.split()[1].isdigit():
            return int(page_key.split()[1])
        else:
            return 0
    
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
        start_time = time.time()
        print(f"\n=== Starting First Two Pages Generation ===")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        structured_urls = convert_dict_to_structured(image_urls, full_image_urls)
        
        # Calculate and log total execution time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n=== First Two Pages Generation Complete ===")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        return GenerateImageResponse(image_urls=structured_urls)
    
    async def generate_images(
        self,
        prompts: Dict[str, str],
        page_connections: Optional[Dict[str, str]],
        reference_images: Optional[List[UploadFile]],
        gender: str,
        age: int,
        image_style: str,
        language: str = "English",
        coverpage: str = "no",
        sequential: str = "no",
        story: Optional[Dict[str, str]] = None,
        page_0_url: Optional[str] = None,
        page_1_url: Optional[str] = None
    ) -> GenerateImageResponse:
        """Generate images for all pages or skip cover/page 1 if they exist"""
        start_time = time.time()
        print(f"\n=== Starting Full Image Generation ===")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Generate unique session ID for this request
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        # Generate unique book UUID for S3 directory structure
        book_uuid = str(uuid.uuid4())
        
        # Determine which pages to generate
        if coverpage.lower() == "yes":
            # Skip page 0 and page 1 - they already exist
            pages_to_generate = {k: v for k, v in prompts.items() if k not in ['page 0', 'page 1']}
            story_to_generate = {k: v for k, v in story.items() if k not in ['page 0', 'page 1']} if story else {}
        else:
            # Generate all pages including page 0 and page 1
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
            dpi=300,
            language=language
        )
        
        # Convert dict to structured format
        structured_urls = convert_dict_to_structured(image_urls, full_image_urls)
        
        # Generate PDF with all pages
        pdf_url = generate_pdf(
            image_bytes=image_bytes,
            book_uuid=book_uuid,
            session_id=session_id,
            upload_to_s3=True
        )
        
        # Calculate and log total execution time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n=== Full Image Generation Complete ===")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
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
        dpi: int = 300,
        language: str= "English"
    ) -> tuple[Dict[str, str], Dict[str, str], Dict[str, bytes]]:
        """Generate images for specified pages using SeeDream API with async parallel text insertion
        Returns: (image_urls, full_image_urls, image_bytes)"""
        # Create shared results dictionary for text insertion
        results_dict = {
            'image_urls': {},
            'full_image_urls': {},
            'image_bytes': {}
        }
        
        # Import text insertion function
        from app.utils.text_insertion_worker import process_single_page_text
        
        print(f"[Processing] Using direct parallel text insertion (no queue)")
        
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
                        results_dict['image_urls']['page 0'] = upload_result['url']
                        results_dict['image_bytes']['page 0'] = page_0_bytes
                        print(f"Uploaded page 0 to S3: {upload_result['url']}")
                    else:
                        print(f"Failed to upload page 0 to S3: {upload_result['message']}")
            except Exception as e:
                print(f"Error processing page 0 URL: {str(e)}")
        
        # If page_1_url is provided, read the split images from local disk (they already have text!)
        if page_1_url and should_split:
            try:
                # Extract the directory path from the full image URL
                if page_1_url.startswith('http'):
                    from urllib.parse import urlparse
                    parsed = urlparse(page_1_url)
                    file_path = parsed.path.lstrip('/')
                else:
                    file_path = page_1_url.lstrip('/')
                
                # The full image is at: uploads/generated_images/{session_id}_image_1.png
                # The split images are at: uploads/generated_images/splitted/{session_id}_page_1.png and page_2.png
                
                # Extract session_id from the file path
                filename = os.path.basename(file_path)
                # Format: {session_id}_image_1.png
                split_session_id = filename.replace('_image_1.png', '')
                
                # Construct paths to the split images (these already have text!)
                left_filename_local = f"uploads/generated_images/splitted/{split_session_id}_page_1.png"
                right_filename_local = f"uploads/generated_images/splitted/{split_session_id}_page_2.png"
                
                # Check if split images exist
                if os.path.exists(left_filename_local) and os.path.exists(right_filename_local):
                    print(f"[Split Images] Found existing split images with text:")
                    print(f"  - Left: {left_filename_local}")
                    print(f"  - Right: {right_filename_local}")
                    
                    # Read the split images (these already contain text)
                    with open(left_filename_local, 'rb') as f:
                        left_bytes_for_pdf = f.read()
                    with open(right_filename_local, 'rb') as f:
                        right_bytes_for_pdf = f.read()
                    
                    # Also read the full image for reference
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            page_1_bytes = f.read()
                        generated_images_bytes['page 1'] = page_1_bytes
                    
                    if upload_to_s3:
                        # Upload full page 1 image to S3
                        if os.path.exists(file_path):
                            full_page_1_buffer = io.BytesIO(page_1_bytes)
                            full_page_1_object_name = f"facetoon/{book_uuid}/full/image_1.png"
                            full_page_1_result = upload_file_object_to_s3(full_page_1_buffer, object_name=full_page_1_object_name)
                            
                            if full_page_1_result['success']:
                                results_dict['full_image_urls']['page 1'] = full_page_1_result['url']
                                print(f"Uploaded full page 1 to S3: {full_page_1_result['url']}")
                        
                        # Upload the existing split images (with text) to S3
                        left_buffer = io.BytesIO(left_bytes_for_pdf)
                        right_buffer = io.BytesIO(right_bytes_for_pdf)
                        
                        left_object_name = f"facetoon/{book_uuid}/splitted/page_1.png"
                        right_object_name = f"facetoon/{book_uuid}/splitted/page_2.png"
                        
                        left_result = upload_file_object_to_s3(left_buffer, object_name=left_object_name)
                        right_result = upload_file_object_to_s3(right_buffer, object_name=right_object_name)
                        
                        if left_result['success'] and right_result['success']:
                            results_dict['image_urls']['page 1'] = left_result['url']
                            results_dict['image_urls']['page 2'] = right_result['url']
                            results_dict['image_bytes']['page 1'] = left_bytes_for_pdf
                            results_dict['image_bytes']['page 2'] = right_bytes_for_pdf
                            print(f"Uploaded split page 1 (with text) to S3")
                            print(f"  - page 1: {left_result['url']}")
                            print(f"  - page 2: {right_result['url']}")
                        else:
                            print(f"Failed to upload split pages to S3")
                    else:
                        # Already stored locally, just set the URLs
                        base_url = os.getenv('domain') or os.getenv('BASE_URL')
                        base_url = base_url.rstrip('/')
                        
                        results_dict['image_urls']['page 1'] = f"{base_url}/{left_filename_local}"
                        results_dict['image_urls']['page 2'] = f"{base_url}/{right_filename_local}"
                        results_dict['image_bytes']['page 1'] = left_bytes_for_pdf
                        results_dict['image_bytes']['page 2'] = right_bytes_for_pdf
                        
                        if os.path.exists(file_path):
                            results_dict['full_image_urls']['page 1'] = f"{base_url}/{file_path}"
                        
                        print(f"Using existing split page 1 images (with text)")
                        print(f"  - page 1: {left_filename_local}")
                        print(f"  - page 2: {right_filename_local}")
                    
                    page_counter_start = 1  # Next generated image will be image 1 (pages 3-4)
                else:
                    print(f"Warning: Split images not found at expected locations:")
                    print(f"  - {left_filename_local}")
                    print(f"  - {right_filename_local}")
                    print(f"Will read full image and re-split (text will be lost)")
                    # Fall through to old logic below
            except Exception as e:
                import traceback
                print(f"Error loading split page 1 images: {str(e)}")
                traceback.print_exc()
                print(f"Will try to read full image and re-split")
        
        page_counter = page_counter_start
        
        # NEW LOGIC: Ignore sequential/parallel parameters
        # Step 1: Generate cover (page 0) first if it exists
        # Step 2: Generate all remaining pages in batches using cover as style reference
        
        print(f"\n=== New Generation Strategy ===")
        print(f"Total pages to generate: {len(pages)}")
        print(f"Batch size: {self.parallel_batch_size}")
        
        cover_page_bytes = None
        remaining_pages = {}
        
        # Separate cover from other pages
        for page_key, prompt in pages.items():
            if page_key == 'page 0':
                print(f"Cover page found, will generate first")
            else:
                remaining_pages[page_key] = prompt
        
        # Step 1: Generate cover page first if it exists
        if 'page 0' in pages:
            print(f"\n[Step 1/2] Generating cover page...")
            cover_prompt = pages['page 0']
            
            cover_bytes, is_single = await asyncio.to_thread(
                self._generate_single_image,
                cover_prompt,
                reference_images_bytes_list,  # Cover uses reference image
                None,  # No style reference for cover
                gender,
                age,
                image_style,
                'page 0',
                session_id,
                reference_page_bytes=None,
                page_number=0
            )
            
            # Store cover bytes for use as style reference
            cover_page_bytes = cover_bytes
            
            # Process cover text insertion immediately (non-blocking)
            cover_task = asyncio.create_task(
                process_single_page_text(
                    'page 0', cover_bytes, 0, is_single,
                    story or {}, font_size, text_color,
                    dpi, session_id, upload_to_s3, book_uuid, language
                )
            )
            print(f"✓ Cover generated, text insertion started in background")
        
        # Step 2: Generate remaining pages in batches
        if remaining_pages:
            print(f"\n[Step 2/2] Generating {len(remaining_pages)} remaining pages in batches of {self.parallel_batch_size}")
            
            async def generate_single_page(page_key, prompt, page_num_for_split):
                """Generate a single page image"""
                try:
                    image_bytes, is_single_page = await asyncio.to_thread(
                        self._generate_single_image,
                        prompt,
                        reference_images_bytes_list,  # All pages use reference image
                        None,  # No URL reference
                        gender,
                        age,
                        image_style,
                        page_key,
                        session_id,
                        reference_page_bytes=cover_page_bytes,  # Use cover as style reference
                        page_number=page_num_for_split
                    )
                    return (page_key, image_bytes, page_num_for_split, is_single_page)
                except Exception as e:
                    print(f"Error generating {page_key}: {str(e)}")
                    raise
            
            # Prepare all pages with their page numbers
            pages_with_numbers = []
            current_page_counter = page_counter
            
            for page_key, prompt in remaining_pages.items():
                # Determine page_number for splitting calculation
                page_num = self._get_page_number(page_key)
                is_coloring_page = page_num == 12 or page_num == 13
                
                page_num_for_split = current_page_counter
                pages_with_numbers.append((page_key, prompt, page_num_for_split))
                
                # Only increment counter for pages that will be split (not coloring pages)
                if not is_coloring_page:
                    current_page_counter += 1
            
            # Generate in batches with 2-second stagger (start next batch 2s after previous starts)
            total_batches = (len(pages_with_numbers) + self.parallel_batch_size - 1) // self.parallel_batch_size
            
            all_text_tasks = []  # Track all text insertion tasks
            
            async def generate_batch_with_delay(batch_idx, delay_seconds):
                """Generate a batch after a delay, then immediately start text insertion for that batch"""
                if delay_seconds > 0:
                    print(f"\n[Batch Scheduler] Waiting {delay_seconds}s before starting next batch...")
                    await asyncio.sleep(delay_seconds)
                
                batch = pages_with_numbers[batch_idx:batch_idx + self.parallel_batch_size]
                batch_num = (batch_idx // self.parallel_batch_size) + 1
                
                batch_start_time = time.time()
                print(f"\n[Batch {batch_num}/{total_batches}] Starting generation of {len(batch)} pages: {[p[0] for p in batch]}")
                
                # Generate batch concurrently
                batch_tasks = [generate_single_page(pk, pr, pn) for pk, pr, pn in batch]
                batch_results = await asyncio.gather(*batch_tasks)
                
                batch_gen_time = time.time() - batch_start_time
                print(f"[Batch {batch_num}/{total_batches}] Generation complete in {batch_gen_time:.2f}s")
                
                # Immediately process text insertion for ALL pages in this batch IN PARALLEL
                print(f"[Batch {batch_num}/{total_batches}] Starting text insertion for {len(batch_results)} pages in parallel")
                text_tasks = []
                for page_key, image_bytes, page_num_for_split, is_single_page in batch_results:
                    task = asyncio.create_task(
                        process_single_page_text(
                            page_key, image_bytes, page_num_for_split, is_single_page,
                            story or {}, font_size, text_color,
                            dpi, session_id, upload_to_s3, book_uuid, language
                        )
                    )
                    text_tasks.append(task)
                
                # Wait for this batch's text insertion to complete and collect results
                text_results = await asyncio.gather(*text_tasks)
                
                # Store results in shared dictionary
                for result in text_results:
                    if result:
                        page_key = result['pageKey']
                        if result.get('leftPageUrl'):
                            # Split page - use returned page numbers
                            left_page_num = result['leftPageNum']
                            right_page_num = result['rightPageNum']
                            results_dict['image_urls'][f'page {left_page_num}'] = result['leftPageUrl']
                            results_dict['image_urls'][f'page {right_page_num}'] = result['rightPageUrl']
                            # Store bytes if available
                            if result.get('leftPageBytes'):
                                results_dict['image_bytes'][f'page {left_page_num}'] = result['leftPageBytes']
                            if result.get('rightPageBytes'):
                                results_dict['image_bytes'][f'page {right_page_num}'] = result['rightPageBytes']
                        else:
                            # Single page (cover, coloring, back cover)
                            results_dict['image_urls'][page_key] = result['fullPageUrl']
                            # Store bytes for PDF if this is a numbered page
                            if result.get('singlePageBytes'):
                                results_dict['image_bytes'][page_key] = result['singlePageBytes']
                        results_dict['full_image_urls'][page_key] = result['fullPageUrl']
                
                print(f"✓ Batch {batch_num}/{total_batches} text insertion complete")
            
            # Start all batches with staggered delays (each batch starts 2s after previous)
            batch_tasks = []
            for batch_idx in range(0, len(pages_with_numbers), self.parallel_batch_size):
                delay = (batch_idx // self.parallel_batch_size) * 2  # 0s, 2s, 4s, 6s, etc.
                task = asyncio.create_task(generate_batch_with_delay(batch_idx, delay))
                batch_tasks.append(task)
            
            # Wait for all batches to complete (they run with staggered starts)
            print(f"\n[Batch Scheduler] Starting {len(batch_tasks)} batches with 2s stagger")
            await asyncio.gather(*batch_tasks)
            print(f"\n[Batch Scheduler] All batches generation and text insertion complete")
        
        print(f"\n=== All image generation complete ===")
        print(f"Generated: {len(pages)} pages total")
        
        # Wait for cover text insertion if it was started
        if 'page 0' in pages:
            cover_result = await cover_task
            if cover_result:
                page_key = cover_result['pageKey']
                if cover_result.get('leftPageUrl'):
                    # Split page - use returned page numbers
                    left_page_num = cover_result['leftPageNum']
                    right_page_num = cover_result['rightPageNum']
                    results_dict['image_urls'][f'page {left_page_num}'] = cover_result['leftPageUrl']
                    results_dict['image_urls'][f'page {right_page_num}'] = cover_result['rightPageUrl']
                    # Store bytes if available
                    if cover_result.get('leftPageBytes'):
                        results_dict['image_bytes'][f'page {left_page_num}'] = cover_result['leftPageBytes']
                    if cover_result.get('rightPageBytes'):
                        results_dict['image_bytes'][f'page {right_page_num}'] = cover_result['rightPageBytes']
                else:
                    # Single page
                    results_dict['image_urls'][page_key] = cover_result['fullPageUrl']
                    # Store bytes for PDF
                    if cover_result.get('singlePageBytes'):
                        results_dict['image_bytes'][page_key] = cover_result['singlePageBytes']
                results_dict['full_image_urls'][page_key] = cover_result['fullPageUrl']
            print(f"Cover text insertion complete")
        
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
ABSOLUTELY NO TEXT, LETTERS, WORDS, TITLES, LABELS, OR ANY WRITTEN CHARACTERS IN THE IMAGE. Pure illustration only.
""".strip()
            else:
                # Check if this is a coloring page (pages 12 or 13)
                page_num = self._get_page_number(page_key)
                is_coloring_page = page_num == 12 or page_num == 13
                
                # Story pages - no text generation, image only
                enhanced_prompt = f"""
Children's storybook illustration in {image_style} style.
Main character: {age}-year-old {gender} child continuing from the previous page.
{prompt}
Composition suitable for a children's storybook with space for text placement on the left side.
CRITICAL COMPOSITION: This image will be split vertically down the middle into two pages. DO NOT place the character's face or any important facial features in the center of the image. Position the character primarily on the LEFT side or RIGHT side of the composition, never centered. Keep the character's face at least 25% away from the center vertical line to avoid splitting facial features. Background elements can span across, but the character should be clearly positioned to one side.
Style: Professional children's book illustration, vibrant colors, high quality, storybook art, child-friendly, whimsical and engaging.
CRITICAL: Maintain EXACT character appearance from the style reference - same facial features, eyebrows, eye shape, nose, mouth, hair color, hair style, and skin tone. If clothing colors or patterns are specified in the prompt, follow them precisely without variation.
Focus on: implementing the exact scene, actions, setting, and clothing details as described while preserving all character appearance characteristics.
ABSOLUTELY NO TEXT, LETTERS, WORDS, SIGNS, LABELS, CAPTIONS, OR ANY WRITTEN CHARACTERS ANYWHERE IN THE IMAGE. The image must be purely visual with no typography of any kind.
Negative prompt: No text, no letters, no words, no signs, no labels, no captions, no written language, no typography, no alphabet characters, no numbers in text form. No changes to the character's face structure, facial proportions, eyebrow thickness or shape, eye color or shape, nose shape, mouth shape, hair color, hair style, or skin tone. Do not modify clothing colors from the prompt description. No artistic reinterpretation of the character's established appearance. Do not center the character's face in the middle of the image.
""".strip()
            
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Prepare payload with appropriate dimensions
            # Pages 0, 12, 13, 14: 8.5"x8.5" at 300 DPI = 2550x2550 pixels
            # Other pages: 16:9 aspect ratio (5120x2880) to later resize to 5100x2550 (17"x8.5" at 300 DPI)
            # Note: Cannot use 'size' and 'width'/'height' together per SeeDream docs
            page_num = self._get_page_number(page_key)
            if page_num == 0 or page_num == 12 or page_num == 13 or page_num == 14:
                # Square format for cover, coloring pages, and back cover
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
            image_bytes, is_single_page = resize_image_to_print_size(image_url, page_key)
            
            # Return bytes and is_single_page flag
            return image_bytes, is_single_page
            
        except Exception as e:
            print(f"Error calling SeeDream API: {str(e)}")
            raise

        