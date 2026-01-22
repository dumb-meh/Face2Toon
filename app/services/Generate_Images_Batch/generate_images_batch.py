import os
import json
import requests
import time
import traceback
from dotenv import load_dotenv
from .generate_images_batch_schema import GenerateImageResponse, PageImageUrls
from typing import Dict, Optional, List
from fastapi import UploadFile
import asyncio
import io
import base64
import uuid
from datetime import datetime
from app.utils.image_processing import resize_image_to_print_size, convert_dict_to_structured
from app.utils.pdf_generator import generate_pdf

load_dotenv()

class GenerateImages:
    def __init__(self):
        self.api_key = os.getenv("ARK_API_KEY")
        print(f"DEBUG: ARK_API_KEY loaded: {bool(self.api_key)} (length: {len(self.api_key) if self.api_key else 0})")
        self.model = "seedream-4-5-251128"  # Batch generation model
        self.base_url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
    
    @staticmethod
    def _get_page_number(page_key: str) -> int:
        """Extract page number from page_key, mapping 'page last page' to 14"""
        if page_key == 'page last page':
            return 14  # Back cover page
        elif page_key.startswith('page ') and page_key.split()[1].isdigit():
            return int(page_key.split()[1])
        else:
            return 0
    
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
        """Generate ALL images at once using Seedream batch API (single API call for all 15 pages)"""
        start_time = time.time()
        print(f"\n=== Starting Batch Image Generation (ALL PAGES IN SINGLE API CALL) ===")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Generate unique session ID and book UUID
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        book_uuid = str(uuid.uuid4())
        
        # Read all reference images
        reference_images_bytes_list = []
        if reference_images:
            for ref_img in reference_images:
                img_bytes = ref_img.file.read()
                ref_img.file.seek(0)
                reference_images_bytes_list.append(img_bytes)
        
        # Handle coverpage="yes" - load existing cover and page 1, use them as references instead of original photo
        pages_to_generate = prompts
        
        if coverpage.lower() == "yes":
            print(f"\n[Coverpage Mode] Cover and page 1 already exist, will use them as references")
            
            # Filter out page 0 and page 1 from generation
            pages_to_generate = {k: v for k, v in prompts.items() if k not in ['page 0', 'page 1']}
            story_to_generate = {k: v for k, v in story.items() if k not in ['page 0', 'page 1']} if story else story
            story = story_to_generate
            
            # IMPORTANT: Replace original child photo with page 0 and page 1 as references
            reference_images_bytes_list = []
            
            # Load existing cover page image (page 0)
            if page_0_url:
                try:
                    from urllib.parse import urlparse
                    if page_0_url.startswith('http'):
                        parsed = urlparse(page_0_url)
                        file_path = parsed.path.lstrip('/')
                    else:
                        file_path = page_0_url.lstrip('/')
                    
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            page_0_bytes = f.read()
                        reference_images_bytes_list.append(page_0_bytes)
                        print(f"[Coverpage Mode] Added page 0 as reference image")
                    else:
                        print(f"[Coverpage Mode] WARNING: Cover image not found at {file_path}")
                except Exception as e:
                    print(f"[Coverpage Mode] Error loading cover image: {str(e)}")
            
            # Load existing page 1 full image
            if page_1_url:
                try:
                    from urllib.parse import urlparse
                    if page_1_url.startswith('http'):
                        parsed = urlparse(page_1_url)
                        file_path = parsed.path.lstrip('/')
                    else:
                        file_path = page_1_url.lstrip('/')
                    
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            page_1_bytes = f.read()
                        reference_images_bytes_list.append(page_1_bytes)
                        print(f"[Coverpage Mode] Added page 1 as reference image")
                    else:
                        print(f"[Coverpage Mode] WARNING: Page 1 image not found at {file_path}")
                except Exception as e:
                    print(f"[Coverpage Mode] Error loading page 1 image: {str(e)}")
            
            print(f"[Coverpage Mode] Using {len(reference_images_bytes_list)} reference images (page 0 and page 1)")
        else:
            print(f"\n[Full Generation Mode] Generating all {len(prompts)} pages, using {len(reference_images_bytes_list)} original child photo(s) as reference")
        
        # Separate coloring pages from main batch - they will be generated separately using single image method
        main_prompts = {}
        coloring_prompts = {}
        for page_key, prompt in pages_to_generate.items():
            page_num = self._get_page_number(page_key)
            if page_num == 12 or page_num == 13:
                coloring_prompts[page_key] = prompt
            else:
                main_prompts[page_key] = prompt
        
        print(f"\n[Strategy] Main batch: {len(main_prompts)} pages | Coloring pages: {len(coloring_prompts)} (will use single image generation)")
        
        # Step 1: Generate main batch only (no coloring pages in batch)
        print(f"\n[Step 1/4] Generating main batch:")
        print(f"  - Main batch: {len(main_prompts)} pages (cover/story/back cover)")
        print(f"  - Coloring pages will be generated separately after main batch")
        
        # Step 1: Generate main batch only (no coloring pages in batch)
        print(f"\n[Step 1/4] Generating main batch:")
        print(f"  - Main batch: {len(main_prompts)} pages (cover/story/back cover)")
        print(f"  - Coloring pages will be generated separately after main batch")
        
        # Generate only main batch
        main_task = asyncio.to_thread(
            self._generate_all_images_batch,
            main_prompts,
            reference_images_bytes_list,
            gender,
            age,
            image_style
        )
        
        # Wait for main batch
        print(f"[Main Batch] Waiting for {len(main_prompts)} images...")
        main_image_urls = await main_task
        print(f"✓ Main batch complete: {len(main_image_urls)} images received")
        
        # Step 2: Download and resize main batch images (while coloring pages may still be generating)
        print(f"\n[Step 2/4] Downloading and resizing main batch images...")
        main_page_keys = list(main_prompts.keys())
        # Step 2: Download and resize main batch images (while coloring pages may still be generating)
        print(f"\n[Step 2/4] Downloading and resizing main batch images...")
        main_page_keys = list(main_prompts.keys())
        
        async def download_and_resize(page_key, image_url):
            """Download and resize a single image"""
            image_bytes, is_single_page = await asyncio.to_thread(
                resize_image_to_print_size,
                image_url,
                page_key
            )
            return (page_key, image_bytes, is_single_page)
        
        # Download main batch
        main_download_tasks = [
            download_and_resize(main_page_keys[i], main_image_urls[i])
            for i in range(len(main_image_urls))
        ]
        main_download_results = await asyncio.gather(*main_download_tasks)
        print(f"✓ Downloaded and resized main batch images")
        
        # Step 2.5: Generate coloring pages using single image method with only cover as reference
        coloring_download_results = []
        if coloring_prompts:
            print(f"\n[Coloring Pages] Generating {len(coloring_prompts)} coloring pages using single image method...")
            
            # Get cover image bytes for reference
            cover_ref_bytes = None
            if coverpage.lower() == "yes" and page_0_url:
                # Load from existing page 0
                try:
                    from urllib.parse import urlparse
                    if page_0_url.startswith('http'):
                        parsed = urlparse(page_0_url)
                        file_path = parsed.path.lstrip('/')
                    else:
                        file_path = page_0_url.lstrip('/')
                    
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            cover_ref_bytes = f.read()
                        print(f"[Coloring Pages] Loaded existing cover as reference")
                except Exception as e:
                    print(f"[Coloring Pages] Error loading cover: {str(e)}")
            else:
                # Use first generated image (page 0) from main batch
                if main_download_results and main_download_results[0][0] == 'page 0':
                    cover_ref_bytes = main_download_results[0][1]
                    print(f"[Coloring Pages] Using generated cover as reference")
            
            # Generate coloring pages in parallel using single image method
            async def generate_single_coloring_page(page_key, prompt):
                """Generate a single coloring page"""
                try:
                    image_bytes, is_single_page = await asyncio.to_thread(
                        self._generate_single_image,
                        prompt,
                        [cover_ref_bytes] if cover_ref_bytes else [],
                        gender,
                        age,
                        image_style,
                        page_key
                    )
                    return (page_key, image_bytes, is_single_page)
                except Exception as e:
                    print(f"[Coloring Pages] Error generating {page_key}: {str(e)}")
                    raise
            
            coloring_tasks = [
                generate_single_coloring_page(page_key, prompt)
                for page_key, prompt in coloring_prompts.items()
            ]
            coloring_download_results = await asyncio.gather(*coloring_tasks)
            print(f"✓ Coloring pages generated: {len(coloring_download_results)} images")
        
        # Combine all download results
        download_results = main_download_results + coloring_download_results
        
        # Step 3: Insert text and split images in parallel
        print(f"\n[Step 3/4] Inserting text and splitting images...")
        from app.utils.text_insertion_worker import process_single_page_text
        
        results_dict = {
            'image_urls': {},
            'full_image_urls': {},
            'image_bytes': {}
        }
        
        # If coverpage="yes", handle existing page 0 and page 1 first
        if coverpage.lower() == "yes":
            # Upload page 0 to S3 if exists
            if page_0_url:
                try:
                    from urllib.parse import urlparse
                    if page_0_url.startswith('http'):
                        parsed = urlparse(page_0_url)
                        file_path = parsed.path.lstrip('/')
                    else:
                        file_path = page_0_url.lstrip('/')
                    
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            page_0_bytes = f.read()
                        
                        # Upload to S3
                        page_0_object_name = f"facetoon/{book_uuid}/page_0.png"
                        page_0_buffer = io.BytesIO(page_0_bytes)
                        from app.utils.upload_to_bucket import upload_file_object_to_s3
                        upload_result = upload_file_object_to_s3(page_0_buffer, object_name=page_0_object_name)
                        
                        if upload_result['success']:
                            results_dict['image_urls']['page 0'] = upload_result['url']
                            results_dict['full_image_urls']['page 0'] = upload_result['url']
                            results_dict['image_bytes']['page 0'] = page_0_bytes
                            print(f"[Coverpage Mode] Uploaded existing page 0 to S3")
                except Exception as e:
                    print(f"[Coverpage Mode] Error uploading page 0: {str(e)}")
            
            # Handle existing page 1 splits
            if page_1_url:
                try:
                    from urllib.parse import urlparse
                    from app.utils.upload_to_bucket import upload_file_object_to_s3
                    
                    if page_1_url.startswith('http'):
                        parsed = urlparse(page_1_url)
                        file_path = parsed.path.lstrip('/')
                    else:
                        file_path = page_1_url.lstrip('/')
                    
                    filename = os.path.basename(file_path)
                    split_session_id = filename.replace('_image_1.png', '')
                    
                    left_filename = f"uploads/generated_images/splitted/{split_session_id}_page_1.png"
                    right_filename = f"uploads/generated_images/splitted/{split_session_id}_page_2.png"
                    
                    if os.path.exists(left_filename) and os.path.exists(right_filename):
                        # Read split images
                        with open(left_filename, 'rb') as f:
                            left_bytes = f.read()
                        with open(right_filename, 'rb') as f:
                            right_bytes = f.read()
                        
                        # Upload splits to S3
                        left_object = f"facetoon/{book_uuid}/splitted/page_1.png"
                        right_object = f"facetoon/{book_uuid}/splitted/page_2.png"
                        
                        left_buffer = io.BytesIO(left_bytes)
                        right_buffer = io.BytesIO(right_bytes)
                        
                        left_result = upload_file_object_to_s3(left_buffer, object_name=left_object)
                        right_result = upload_file_object_to_s3(right_buffer, object_name=right_object)
                        
                        if left_result['success'] and right_result['success']:
                            results_dict['image_urls']['page 1'] = left_result['url']
                            results_dict['image_urls']['page 2'] = right_result['url']
                            results_dict['image_bytes']['page 1'] = left_bytes
                            results_dict['image_bytes']['page 2'] = right_bytes
                            
                            # Also read full image for full_image_urls
                            if os.path.exists(file_path):
                                with open(file_path, 'rb') as f:
                                    full_bytes = f.read()
                                full_object = f"facetoon/{book_uuid}/full/image_1.png"
                                full_buffer = io.BytesIO(full_bytes)
                                full_result = upload_file_object_to_s3(full_buffer, object_name=full_object)
                                if full_result['success']:
                                    results_dict['full_image_urls']['page 1'] = full_result['url']
                            
                            print(f"[Coverpage Mode] Uploaded existing page 1 splits to S3")
                except Exception as e:
                    print(f"[Coverpage Mode] Error uploading page 1: {str(e)}")
        
        text_tasks = []
        page_counter = 0 if coverpage.lower() != "yes" else 1  # Start at 1 if skipping page 0 and page 1
        for page_key, image_bytes, is_single_page in download_results:
            page_num = self._get_page_number(page_key)
            is_coloring_page = page_num == 12 or page_num == 13
            
            # Determine page_num_for_split
            if page_key == 'page 0':
                page_num_for_split = 0
            elif is_coloring_page or page_num == 14:
                page_num_for_split = page_counter
            else:
                page_num_for_split = page_counter
                page_counter += 1  # Increment for split pages
            
            task = asyncio.create_task(
                process_single_page_text(
                    page_key, image_bytes, page_num_for_split, is_single_page,
                    story or {}, 100, "white", 300, session_id,
                    True, book_uuid, language
                )
            )
            text_tasks.append(task)
        
        text_results = await asyncio.gather(*text_tasks)
        
        # Store results
        for result in text_results:
            if result:
                page_key = result['pageKey']
                if result.get('leftPageUrl'):
                    # Split page
                    left_page_num = result['leftPageNum']
                    right_page_num = result['rightPageNum']
                    results_dict['image_urls'][f'page {left_page_num}'] = result['leftPageUrl']
                    results_dict['image_urls'][f'page {right_page_num}'] = result['rightPageUrl']
                    if result.get('leftPageBytes'):
                        results_dict['image_bytes'][f'page {left_page_num}'] = result['leftPageBytes']
                    if result.get('rightPageBytes'):
                        results_dict['image_bytes'][f'page {right_page_num}'] = result['rightPageBytes']
                else:
                    # Single page
                    results_dict['image_urls'][page_key] = result['fullPageUrl']
                    if result.get('singlePageBytes'):
                        results_dict['image_bytes'][page_key] = result['singlePageBytes']
                results_dict['full_image_urls'][page_key] = result['fullPageUrl']
        
        print(f"✓ Text insertion and splitting complete")
        
        # Step 4: Generate PDF
        print(f"\n[Step 4/4] Generating PDF...")
        pdf_url = generate_pdf(
            image_bytes=results_dict['image_bytes'],
            book_uuid=book_uuid,
            session_id=session_id,
            upload_to_s3=True
        )
        print(f"✓ PDF generated: {pdf_url}")
        
        # Convert dict to structured format
        structured_urls = convert_dict_to_structured(
            results_dict['image_urls'],
            results_dict['full_image_urls']
        )
        
        # Convert PageImageUrls objects to dictionaries for Pydantic validation
        structured_urls_dicts = [page.model_dump() for page in structured_urls]
        
        # Calculate and log total execution time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n=== Batch Image Generation Complete ===")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        return GenerateImageResponse(image_urls=structured_urls_dicts, pdf_url=pdf_url)
    
    def _generate_all_images_batch(
        self,
        prompts: Dict[str, str],
        reference_images_bytes: List[bytes],
        gender: str,
        age: int,
        image_style: str
    ) -> List[str]:
        """Generate ALL images in a single batch API call using Seedream sequential generation"""
        try:
            # Add age, gender, and brief negative prompting to each prompt (keep short for batch processing)
            enhanced_prompts = {}
            for page_key, prompt in prompts.items():
                page_num = self._get_page_number(page_key)
                
                # Add age, gender, and strict character matching instructions
                if page_num == 12 or page_num == 13:
                    # Coloring pages - emphasize black and white only
                    enhanced_prompts[page_key] = f"{prompt}\n\nCRITICAL: {age}-year-old {gender} child. EXACT appearance from reference image - same hair style, hair length, hair color, facial features, face shape. Child from reference image ONLY.\n\nNegative: realistic photo, colors, shading, gradients, different hairstyle, hair length change, gender change, feminine features for boys, masculine features for girls."
                else:
                    # Regular pages - prevent photorealism, text, and appearance changes
                    enhanced_prompts[page_key] = f"{prompt}\n\nCRITICAL: {age}-year-old {gender} child. EXACT appearance from reference image - same hair style, hair length, hair color, facial features, eyebrows, eye shape, face shape. Match reference image EXACTLY.\n\nNegative: realistic photo, photograph, text, letters, centered character, gender change, different hairstyle, hair length change, feminine look for boys, masculine look for girls, must maintain {gender} appearance from reference."
            
            # Convert enhanced prompts dict to JSON string for batch API
            prompts_json = json.dumps(enhanced_prompts)
            
            print(f"[Batch API] Preparing to generate {len(prompts)} images in single call")
            print(f"[Batch API] Model: {self.model}")
            
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Prepare payload - use first reference image
            payload = {
                'model': self.model,
                'prompt': prompts_json,
                'width': 5100,
                'height': 2550,
                'watermark': False,
                'sequential_image_generation': 'auto',
                'sequential_image_generation_options': {
                    'max_images': len(prompts)
                }
            }
            
            # Add reference images (up to 3)
            if reference_images_bytes and len(reference_images_bytes) > 0:
                for idx, img_bytes in enumerate(reference_images_bytes[:3], 1):
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    if idx == 1:
                        payload['reference_image'] = img_base64
                    else:
                        payload[f'reference_image_{idx}'] = img_base64
            
            print(f"[Batch API] Sending request to Seedream...")
            print(f"[Batch API] This may take 5-10 minutes to generate all images...")
            
            # Call Seedream batch API with extended timeout
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=1200  # 20 minutes timeout for batch generation
            )
            
            # Debug: Print response details
            if response.status_code != 200:
                print(f"[Batch API] ERROR - Status Code: {response.status_code}")
                print(f"[Batch API] Response: {response.text}")
                response.raise_for_status()
            
            result = response.json()
            print(f"[Batch API] Success! Status Code: {response.status_code}")
            
            # Extract image URLs from response
            # Response format: {"data": [{"url": "...", "size": "2048x2048"}, ...]}
            if 'data' not in result:
                raise Exception(f"No 'data' field in API response: {result}")
            
            image_urls = [item['url'] for item in result['data']]
            print(f"[Batch API] Extracted {len(image_urls)} image URLs")
            
            if len(image_urls) != len(prompts):
                print(f"[Batch API] WARNING: Expected {len(prompts)} images, got {len(image_urls)}")
            
            return image_urls
            
        except Exception as e:
            print(f"[Batch API] Error during batch generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_single_image(
        self,
        prompt: str,
        reference_images_bytes: List[bytes],
        gender: str,
        age: int,
        image_style: str,
        page_key: str
    ) -> tuple[bytes, bool]:
        """Generate a single coloring page using SeeDream API (not batch method)"""
        try:
            print(f"[Single Image] Generating {page_key}...")
            
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Add negative prompt for coloring pages with strict character matching
            enhanced_prompt = f"{prompt}\n\nCRITICAL: {age}-year-old {gender} child. EXACT appearance from reference image - same hair style, hair length, hair color, facial features, face shape. Child from reference image ONLY.\n\nNegative: realistic photo, colors, shading, gradients, different hairstyle, hair length change, gender change, feminine features for boys, masculine features for girls."
            
            # Prepare payload - coloring pages are square
            payload = {
                'model': self.model,
                'prompt': enhanced_prompt,
                'width': 2550,
                'height': 2550,
                'watermark': False
            }
            
            # Add reference image (only cover)
            if reference_images_bytes and len(reference_images_bytes) > 0:
                img_base64 = base64.b64encode(reference_images_bytes[0]).decode('utf-8')
                payload['reference_image'] = img_base64
            
            print(f"[Single Image] Sending request for {page_key}...")
            
            # Call SeeDream API
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code != 200:
                print(f"[Single Image] ERROR - Status Code: {response.status_code}")
                print(f"[Single Image] Response: {response.text}")
                response.raise_for_status()
            
            result = response.json()
            
            # Extract image URL
            image_url = result.get('data', [{}])[0].get('url') or result.get('image_url') or result.get('url')
            
            if not image_url:
                raise Exception(f"No image URL in response: {result}")
            
            # Download and resize image
            image_bytes, is_single_page = resize_image_to_print_size(image_url, page_key)
            
            print(f"[Single Image] ✓ {page_key} generated successfully")
            return image_bytes, is_single_page
            
        except Exception as e:
            print(f"[Single Image] Error generating {page_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
