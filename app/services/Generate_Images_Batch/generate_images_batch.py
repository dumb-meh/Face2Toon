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
        
        # Separate coloring pages from main batch (they confuse the model with black/white instructions)
        main_prompts = {}
        coloring_prompts = {}
        for page_key, prompt in prompts.items():
            page_num = self._get_page_number(page_key)
            if page_num == 12 or page_num == 13:
                coloring_prompts[page_key] = prompt
            else:
                main_prompts[page_key] = prompt
        
        print(f"\n[Strategy] Generating {len(main_prompts)} main pages and {len(coloring_prompts)} coloring pages separately")
        
        # Step 1: Generate main batch and coloring pages simultaneously
        print(f"\n[Step 1/4] Starting parallel generation:")
        print(f"  - Main batch: {len(main_prompts)} pages (cover, story, back cover)")
        print(f"  - Coloring pages: {len(coloring_prompts)} pages")
        
        # Start both requests in parallel
        main_task = asyncio.to_thread(
            self._generate_all_images_batch,
            main_prompts,
            reference_images_bytes_list,
            gender,
            age,
            image_style
        )
        
        coloring_task = asyncio.to_thread(
            self._generate_all_images_batch,
            coloring_prompts,
            reference_images_bytes_list,
            gender,
            age,
            image_style
        ) if coloring_prompts else None
        
        # Wait for main batch first (we need it to proceed)
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
        
        # Wait for coloring pages to complete
        coloring_download_results = []
        if coloring_task:
            print(f"\n[Coloring Pages] Waiting for {len(coloring_prompts)} coloring pages...")
            coloring_image_urls = await coloring_task
            print(f"✓ Coloring pages complete: {len(coloring_image_urls)} images received")
            
            # Download coloring pages
            coloring_page_keys = list(coloring_prompts.keys())
            coloring_download_tasks = [
                download_and_resize(coloring_page_keys[i], coloring_image_urls[i])
                for i in range(len(coloring_image_urls))
            ]
            coloring_download_results = await asyncio.gather(*coloring_download_tasks)
            print(f"✓ Downloaded and resized coloring pages")
        
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
        
        text_tasks = []
        page_counter = 0
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
            # Add brief negative prompting to each prompt (keep short for batch processing)
            enhanced_prompts = {}
            for page_key, prompt in prompts.items():
                page_num = self._get_page_number(page_key)
                
                # Add brief negative prompt based on page type
                if page_num == 12 or page_num == 13:
                    # Coloring pages - emphasize black and white only
                    enhanced_prompts[page_key] = f"{prompt}\n\nNegative: realistic photo, colors, shading, gradients."
                else:
                    # Regular pages - prevent photorealism and text
                    enhanced_prompts[page_key] = f"{prompt}\n\nNegative: realistic photo, photograph, text, letters, centered character."
            
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
