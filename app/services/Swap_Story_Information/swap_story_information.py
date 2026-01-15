import os
import io
import requests
import boto3
import asyncio
import uuid
import time
import openai
from datetime import datetime
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from .swap_story_information_schema import SwapStoryInformationRequest, SwapStoryInformationResponse, PageImageUrls
from app.utils.upload_to_bucket import upload_file_object_to_s3
from app.utils.image_analysis import get_text_placement_recommendation
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader

load_dotenv()

class SwapStoryInformation:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
            region_name=os.getenv('S3_REGION', 'eu-north-1')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'mycvconnect')
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def swap_story_information(
        self,
        request: SwapStoryInformationRequest
    ) -> SwapStoryInformationResponse:
        """Main method to swap story information (name and/or language) without regenerating images"""
        start_time = time.time()
        try:
            print(f"\n=== Starting Story Information Swap ===")
            print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Change Type: {request.change_type}")
            print(f"Character Name: {request.character_name}")
            print(f"Story Language: {request.story_language} -> {request.language}")
            print(f"Received {len(request.full_page_urls)} existing images")
            
            # Generate unique book UUID for S3 directory
            book_uuid = str(uuid.uuid4())
            session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Update story text based on change_type
            print(f"\n=== Updating story text ===" )
            updated_story = await self._update_story_based_on_type(
                request.story,
                request.character_name,
                request.story_language,
                request.language,
                request.change_type
            )
            print(f"✓ Updated {len(updated_story)} story entries")
            
            # Results dictionary
            image_urls = {}
            full_image_urls = {}
            image_bytes_for_pdf = {}
            
            # Process pages in parallel
            tasks = []
            
            # Process all pages (cover, story pages, coloring pages)
            for idx, page_url in enumerate(request.full_page_urls):
                page_key = f"page {idx}"
                story_text = updated_story.get(page_key, "")
                
                # Determine if this is a single page (cover or coloring) or double page (story)
                is_single_page = idx == 0 or idx >= 12  # page 0 or pages 12-13
                
                if is_single_page:
                    # Single page - no splitting
                    task = self._process_single_page(
                        page_key=page_key,
                        page_url=page_url,
                        story_text=story_text,
                        book_uuid=book_uuid,
                        language=request.language
                    )
                else:
                    # Double page - needs splitting
                    task = self._process_double_page(
                        page_key=page_key,
                        page_url=page_url,
                        story_text=story_text,
                        book_uuid=book_uuid,
                        page_number=idx - 1,  # For splitting calculation
                        language=request.language
                    )
                
                tasks.append(task)
            
            # Wait for all tasks to complete
            print(f"\nProcessing {len(tasks)} pages in parallel...")
            results = await asyncio.gather(*tasks)
            
            # Collect results
            for result in results:
                if result:
                    image_urls.update(result['image_urls'])
                    full_image_urls.update(result['full_image_urls'])
                    image_bytes_for_pdf.update(result['image_bytes'])
            
            print(f"\n=== Story information swap complete ===")
            print(f"Processed {len(image_urls)} page images")
            
            # Generate PDF
            pdf_url = self._generate_pdf(
                image_bytes=image_bytes_for_pdf,
                book_uuid=book_uuid,
                session_id=session_id
            )
            
            # Convert to structured format
            structured_urls = self._convert_dict_to_structured(image_urls, full_image_urls)
            
            # Calculate and log total execution time
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n=== Story Information Swap Complete ===")
            print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            
            return SwapStoryInformationResponse(image_urls=structured_urls, pdf_url=pdf_url)
            
        except Exception as e:
            print(f"Error in swap_story_information: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _update_story_based_on_type(
        self,
        story_dict: Dict[str, str],
        new_character_name: str,
        current_language: str,
        target_language: str,
        change_type: str
    ) -> Dict[str, str]:
        """Update story text based on change_type: 'language', 'name', or 'both'"""
        try:
            if not story_dict:
                return {}
            
            updated_dict = {}
            
            print(f"[Story Update] Change type: {change_type}")
            print(f"[Story Update] Current language: {current_language}")
            print(f"[Story Update] Target language: {target_language}")
            print(f"[Story Update] New character name: {new_character_name}")
            
            # Process each story entry
            for key, text in story_dict.items():
                if not text or not text.strip():
                    updated_dict[key] = text
                    continue
                
                if change_type == "language":
                    # Only translate, keep names as is
                    llm_prompt = f"""You are translating a children's storybook text to {target_language}.

Target language: {target_language}

Original text (in {current_language}):
{text}

Instructions:
1. Translate the entire text to {target_language}
2. Keep all names EXACTLY as they are - DO NOT translate or change character names
3. Keep the story content, actions, and meaning exactly the same
4. Preserve all pronouns and gender references
5. Return ONLY the translated text, nothing else

Translated text:"""
                
                elif change_type == "name":
                    # Only change name, keep language as is
                    llm_prompt = f"""You are updating a children's storybook text. Replace the old character name with the new name.

New character name: {new_character_name}

Original text:
{text}

Instructions:
1. Replace any character name with "{new_character_name}"
2. Keep everything else EXACTLY the same - same language, same pronouns, same wording
3. DO NOT translate or change the language
4. DO NOT change pronouns or gender references
5. Return ONLY the text with updated name, nothing else

Updated text:"""
                
                elif change_type == "both":
                    # Translate AND change name
                    llm_prompt = f"""You are translating a children's storybook text and updating the character name.

New character name: {new_character_name}
Target language: {target_language}

Original text (in {current_language}):
{text}

Instructions:
1. Replace any character name with "{new_character_name}"
2. Translate the entire text to {target_language}
3. Keep the story content, actions, and meaning exactly the same
4. Preserve pronouns and gender references from the original
5. Return ONLY the translated text with updated name, nothing else

Translated text with updated name:"""
                
                else:
                    # Invalid change_type, return original
                    print(f"[Story Update] Warning: Invalid change_type '{change_type}', keeping original text")
                    updated_dict[key] = text
                    continue
                
                # Call OpenAI
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that updates character names and translates stories while preserving all other details."},
                        {"role": "user", "content": llm_prompt}
                    ],
                    temperature=0.3
                )
                
                updated_text = response.choices[0].message.content.strip()
                updated_dict[key] = updated_text
            
            print(f"[Story Update] ✓ Completed with change_type: {change_type}")
            
            return updated_dict
            
        except Exception as e:
            print(f"Error updating story text: {str(e)}")
            # Fallback: return original text if LLM fails
            return story_dict
    
    async def _process_single_page(
        self,
        page_key: str,
        page_url: str,
        story_text: str,
        book_uuid: str,
        language: str = "English",
        font_size: int = 100,
        text_color: str = "white",
        dpi: int = 300
    ) -> Dict:
        """Process single page (cover or coloring page) - download, add text, upload"""
        try:
            print(f"\n[Single] Processing {page_key}...")
            
            # Download existing image
            print(f"[Single] Downloading image: {page_url}")
            img_response = requests.get(page_url, timeout=60)
            img_response.raise_for_status()
            img_bytes = img_response.content
            
            # Add text to image if story text exists
            img = Image.open(io.BytesIO(img_bytes))
            
            if story_text and story_text.strip():
                print(f"[Single] Adding text to {page_key}...")
                try:
                    text_recommendation = await get_text_placement_recommendation(
                        img_bytes, story_text, font_size
                    )
                    
                    draw = ImageDraw.Draw(img)
                    
                    # Load font based on language
                    from pathlib import Path
                    if language and language.lower() == "arabic":
                        font_path = Path(__file__).resolve().parents[3] / "fonts" / "Playpen_Sans_Arabic" / "PlaypenSansArabic-Regular.ttf"
                    else:
                        font_path = Path(__file__).resolve().parents[3] / "fonts" / "Comic_Relief" / "ComicRelief-Regular.ttf"
                    
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
                    
                    print(f"[Single] Text added successfully")
                except Exception as e:
                    print(f"[Single] Error adding text: {str(e)}")
            
            # Convert image to bytes
            output_buffer = io.BytesIO()
            img.save(output_buffer, format='PNG', dpi=(dpi, dpi))
            output_buffer.seek(0)
            final_img_bytes = output_buffer.read()
            output_buffer.seek(0)
            
            # Upload to S3
            page_num = int(page_key.split()[1])
            object_name = f"facetoon/{book_uuid}/page_{page_num}.png"
            
            upload_result = upload_file_object_to_s3(output_buffer, object_name=object_name)
            
            if not upload_result['success']:
                raise ValueError(f"Failed to upload page: {upload_result['message']}")
            
            print(f"[Single] ✓ Completed {page_key}")
            
            return {
                'image_urls': {
                    page_key: upload_result['url']
                },
                'full_image_urls': {
                    page_key: upload_result['url']
                },
                'image_bytes': {
                    page_key: final_img_bytes
                }
            }
            
        except Exception as e:
            print(f"[Single] Error processing {page_key}: {str(e)}")
            raise
    
    async def _process_double_page(
        self,
        page_key: str,
        page_url: str,
        story_text: str,
        book_uuid: str,
        page_number: int,
        language: str = "English",
        font_size: int = 100,
        text_color: str = "white",
        dpi: int = 300
    ) -> Dict:
        """Process double-width page - download, add text, split, upload"""
        try:
            print(f"\n[Double] Processing {page_key}...")
            
            # Download existing image
            print(f"[Double] Downloading image: {page_url}")
            img_response = requests.get(page_url, timeout=60)
            img_response.raise_for_status()
            img_bytes = img_response.content
            
            # Add text to image if story text exists
            img = Image.open(io.BytesIO(img_bytes))
            
            if story_text and story_text.strip():
                print(f"[Double] Adding text to {page_key}...")
                try:
                    text_recommendation = await get_text_placement_recommendation(
                        img_bytes, story_text, font_size
                    )
                    
                    draw = ImageDraw.Draw(img)
                    
                    # Load font based on language
                    from pathlib import Path
                    if language and language.lower() == "arabic":
                        font_path = Path(__file__).resolve().parents[3] / "fonts" / "Playpen_Sans_Arabic" / "PlaypenSansArabic-Regular.ttf"
                    else:
                        font_path = Path(__file__).resolve().parents[3] / "fonts" / "Comic_Relief" / "ComicRelief-Regular.ttf"
                    
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
                    
                    print(f"[Double] Text added successfully")
                except Exception as e:
                    print(f"[Double] Error adding text: {str(e)}")
            
            # Upload full image
            page_num_str = page_key.replace('page ', '').replace(' ', '_')
            full_object_name = f"facetoon/{book_uuid}/full/image_{page_num_str}.png"
            
            full_img_buffer = io.BytesIO()
            img.save(full_img_buffer, format='PNG', dpi=(dpi, dpi))
            full_img_buffer.seek(0)
            full_img_bytes = full_img_buffer.read()
            full_img_buffer.seek(0)
            
            full_upload_result = upload_file_object_to_s3(full_img_buffer, object_name=full_object_name)
            
            if not full_upload_result['success']:
                raise ValueError(f"Failed to upload full image: {full_upload_result['message']}")
            
            # Split image
            middle_x = img.width // 2
            left_half = img.crop((0, 0, middle_x, img.height))
            right_half = img.crop((middle_x, 0, img.width, img.height))
            
            left_page_num = (page_number * 2) + 1
            right_page_num = (page_number * 2) + 2
            
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
            
            if not left_result['success'] or not right_result['success']:
                raise ValueError("Failed to upload split images")
            
            print(f"[Double] ✓ Completed {page_key} -> pages {left_page_num}, {right_page_num}")
            
            return {
                'image_urls': {
                    f'page {left_page_num}': left_result['url'],
                    f'page {right_page_num}': right_result['url']
                },
                'full_image_urls': {
                    page_key: full_upload_result['url']
                },
                'image_bytes': {
                    f'page {left_page_num}': left_bytes,
                    f'page {right_page_num}': right_bytes
                }
            }
            
        except Exception as e:
            print(f"[Double] Error processing {page_key}: {str(e)}")
            raise
    
    def _convert_dict_to_structured(self, image_urls: Dict[str, str], full_image_urls: Dict[str, str]) -> List[PageImageUrls]:
        """Convert flat dictionary to structured format"""
        structured_pages = []
        processed_pages = set()
        
        sorted_keys = sorted(image_urls.keys(), key=lambda x: int(x.split()[1]) if 'page' in x and x.split()[1].isdigit() else 0)
        
        for key in sorted_keys:
            url = image_urls[key]
            
            if 'page' in key:
                page_num_str = key.split()[1]
                if page_num_str.isdigit():
                    page_num = int(page_num_str)
                    
                    if page_num in processed_pages:
                        continue
                    
                    # Special handling: Pair pages 23-24 as "page 12" (coloring pages)
                    if page_num == 23:
                        right_key = 'page 24'
                        if right_key in image_urls:
                            page_obj = PageImageUrls(
                                name='page 12',
                                fullPageUrl='',
                                leftUrl=url,
                                rightUrl=image_urls[right_key]
                            )
                            structured_pages.append(page_obj)
                            processed_pages.add(23)
                            processed_pages.add(24)
                        continue
                    
                    # Cover page (page 0) - single image
                    if page_num == 0:
                        page_obj = PageImageUrls(
                            name=key,
                            fullPageUrl=url,
                            leftUrl=None,
                            rightUrl=None
                        )
                        structured_pages.append(page_obj)
                        processed_pages.add(page_num)
                        continue
                    
                    # Regular split pairs (1-2, 3-4, etc.) - stops before page 23
                    if page_num % 2 == 1 and page_num < 23:
                        right_page_num = page_num + 1
                        right_key = f'page {right_page_num}'
                        
                        if right_key in image_urls:
                            # Calculate original page number
                            original_page_num = (page_num + 1) // 2
                            original_page_key = f'page {original_page_num}'
                            full_url = full_image_urls.get(original_page_key, '')
                            
                            page_obj = PageImageUrls(
                                name=original_page_key,
                                fullPageUrl=full_url,
                                leftUrl=url,
                                rightUrl=image_urls[right_key]
                            )
                            structured_pages.append(page_obj)
                            processed_pages.add(page_num)
                            processed_pages.add(right_page_num)
        
        return structured_pages
    
    def _generate_pdf(
        self,
        image_bytes: Dict[str, bytes],
        book_uuid: str,
        session_id: str
    ) -> str:
        """Generate PDF book with all pages"""
        try:
            page_size = (8.5 * 72, 8.5 * 72)
            pdf_buffer = io.BytesIO()
            c = pdf_canvas.Canvas(pdf_buffer, pagesize=page_size)
            
            sorted_pages = sorted(
                [(k, v) for k, v in image_bytes.items() if k.startswith('page ')],
                key=lambda x: int(x[0].split()[1])
            )
            
            print(f"\nGenerating PDF with {len(sorted_pages)} pages...")
            
            for page_key, page_bytes in sorted_pages:
                try:
                    img = Image.open(io.BytesIO(page_bytes))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((612, 612), Image.Resampling.LANCZOS)
                    
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    img_reader = ImageReader(img_buffer)
                    
                    c.drawImage(img_reader, 0, 0, width=612, height=612)
                    c.showPage()
                    
                except Exception as e:
                    print(f"Error adding {page_key} to PDF: {str(e)}")
                    continue
            
            c.save()
            pdf_buffer.seek(0)
            
            # Upload to S3
            pdf_object_name = f"facetoon/{book_uuid}/book_{session_id}.pdf"
            upload_result = upload_file_object_to_s3(pdf_buffer, object_name=pdf_object_name)
            
            if upload_result['success']:
                print(f"PDF uploaded to S3: {upload_result['url']}")
                return upload_result['url']
            else:
                print(f"Failed to upload PDF: {upload_result['message']}")
                return None
                
        except Exception as e:
            print(f"Error generating PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
