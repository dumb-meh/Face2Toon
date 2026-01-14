import os
import io
import re
import requests
import base64
import boto3
import asyncio
import uuid
import time
import openai
from datetime import datetime
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from urllib.parse import urlparse
from fastapi import UploadFile
from .swap_character_schema import SwapCharacterResponse, PageImageUrls
from app.utils.upload_to_bucket import upload_file_object_to_s3
from app.utils.image_analysis import get_text_placement_recommendation
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader

load_dotenv()

class SwapCharacter:
    def __init__(self):
        self.api_key = os.getenv("ARK_API_KEY")
        self.model = "seedream-4-0-250828"
        self.base_url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
            region_name=os.getenv('S3_REGION', 'eu-north-1')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'mycvconnect')
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def swap_character(
        self,
        full_page_urls: List[str],
        prompts: Dict[str, str],
        story: Dict[str, str],
        character_name: str,
        gender: str,
        age: int,
        image_style: str,
        reference_images: List[UploadFile]
    ) -> SwapCharacterResponse:
        """Main method to swap character in existing book"""
        start_time = time.time()
        try:
            print(f"\n=== Starting Character Swap ===")
            print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Character: {character_name}, Gender: {gender}, Age: {age}")
            print(f"Style: {image_style}")
            print(f"Received {len(full_page_urls)} existing images")
            print(f"Reference images: {len(reference_images)}")
            
            # Generate unique book UUID for S3 directory
            book_uuid = str(uuid.uuid4())
            session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Read all reference images
            reference_images_bytes_list = []
            for ref_img in reference_images:
                img_bytes = ref_img.file.read()
                ref_img.file.seek(0)
                reference_images_bytes_list.append(img_bytes)
            
            # Update prompts and story with new character details using LLM
            print(f"\n=== Updating prompts and story for new character ===")
            updated_prompts = await self._update_text_with_llm(prompts, character_name, gender, age, is_story=False)
            updated_story = await self._update_text_with_llm(story, character_name, gender, age, is_story=True)
            print(f"✓ Updated {len(updated_prompts)} prompts and {len(updated_story)} story entries")
            
            # Results dictionary
            image_urls = {}
            full_image_urls = {}
            image_bytes_for_pdf = {}
            
            # Process pages in parallel
            tasks = []
            
            # Page 0: Cover page (single square image)
            if len(full_page_urls) > 0:
                cover_url = full_page_urls[0]
                task = self._generate_cover_page(
                    page_url=cover_url,
                    prompt=updated_prompts.get('page 0', ''),
                    character_name=character_name,
                    gender=gender,
                    age=age,
                    image_style=image_style,
                    reference_images_bytes=reference_images_bytes_list,
                    book_uuid=book_uuid
                )
                tasks.append(task)
            
            # Pages 1-11: Swap character using existing images as composition reference
            for idx, page_url in enumerate(full_page_urls[1:], start=1):
                page_key = f"page {idx}"
                page_num = idx
                
                task = self._swap_character_in_page(
                    page_key=page_key,
                    page_url=page_url,
                    prompt=updated_prompts.get(page_key, ""),
                    story_text=updated_story.get(page_key, ""),
                    character_name=character_name,
                    gender=gender,
                    age=age,
                    image_style=image_style,
                    reference_images_bytes=reference_images_bytes_list,
                    book_uuid=book_uuid,
                    page_number=page_num - 1  # For splitting: page 1 -> pages 1-2 (page_number=0)
                )
                tasks.append(task)
            
            # Pages 12-13: Generate coloring pages from scratch
            if 'page 12' in updated_prompts:
                task = self._generate_coloring_page(
                    page_key='page 12',
                    prompt=updated_prompts['page 12'],
                    character_name=character_name,
                    gender=gender,
                    age=age,
                    image_style=image_style,
                    reference_images_bytes=reference_images_bytes_list,
                    book_uuid=book_uuid
                )
                tasks.append(task)
            
            if 'page 13' in updated_prompts:
                task = self._generate_coloring_page(
                    page_key='page 13',
                    prompt=updated_prompts['page 13'],
                    character_name=character_name,
                    gender=gender,
                    age=age,
                    image_style=image_style,
                    reference_images_bytes=reference_images_bytes_list,
                    book_uuid=book_uuid
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
            
            print(f"\n=== Character swap complete ===")
            print(f"Generated {len(image_urls)} page images")
            
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
            print(f"\n=== Character Swap Complete ===")
            print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            
            return SwapCharacterResponse(image_urls=structured_urls, pdf_url=pdf_url)
            
        except Exception as e:
            print(f"Error in swap_character: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _update_text_with_llm(
        self,
        text_dict: Dict[str, str],
        new_character_name: str,
        new_gender: str,
        new_age: int,
        is_story: bool = False
    ) -> Dict[str, str]:
        """Use LLM to update character names and pronouns in prompts/story"""
        try:
            updated_dict = {}
            
            for key, text in text_dict.items():
                if not text or not text.strip():
                    updated_dict[key] = text
                    continue
                
                # Create LLM prompt
                text_type = "story text" if is_story else "image generation prompt"
                pronoun_guide = "he/him/his" if new_gender.lower() == "male" else "she/her/hers"
                
                llm_prompt = f"""You are updating a children's storybook {text_type}. Replace the old character's name and pronouns with new ones.

New character details:
- Name: {new_character_name}
- Gender: {new_gender}
- Age: {new_age} years old
- Pronouns: {pronoun_guide}

Original {text_type}:
{text}

Instructions:
1. Replace any character name with "{new_character_name}"
2. Update all pronouns to match {pronoun_guide}
3. Keep everything else exactly the same - same story, same scene, same actions
4. Return ONLY the updated text, nothing else

Updated {text_type}:"""
                
                # Call OpenAI
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that updates character names and pronouns in text while keeping everything else unchanged."},
                        {"role": "user", "content": llm_prompt}
                    ],
                    temperature=0.3
                )
                
                updated_text = response.choices[0].message.content.strip()
                updated_dict[key] = updated_text
                
            return updated_dict
            
        except Exception as e:
            print(f"Error updating text with LLM: {str(e)}")
            # Fallback: return original text if LLM fails
            return text_dict
    
    async def _generate_cover_page(
        self,
        page_url: str,
        prompt: str,
        character_name: str,
        gender: str,
        age: int,
        image_style: str,
        reference_images_bytes: List[bytes],
        book_uuid: str
    ) -> Dict:
        """Generate cover page (page 0) - single square image"""
        try:
            print(f"\n[Cover] Generating cover page...")
            
            # Download existing cover image
            print(f"[Cover] Downloading existing cover: {page_url}")
            existing_img_response = requests.get(page_url, timeout=60)
            existing_img_response.raise_for_status()
            existing_img_bytes = existing_img_response.content
            
            # Create enhanced prompt
            enhanced_prompt = f"""
Children's storybook cover page in {image_style} style.
Main character: {age}-year-old {gender} child named {character_name} matching the reference image exactly.
{prompt}
Maintain the exact same composition, layout, and design from the style reference.
CRITICAL: Only change the character's appearance to match the new reference images.
ABSOLUTELY NO TEXT, LETTERS, WORDS, OR ANY WRITTEN CHARACTERS IN THE IMAGE.
""".strip()
            
            # Generate new cover with character swap (square format: 2550x2550)
            img_bytes = await asyncio.to_thread(
                self._generate_swapped_image,
                enhanced_prompt,
                reference_images_bytes,
                existing_img_bytes,
                gender,
                age,
                image_style,
                width=2550,
                height=2550
            )
            
            # Upload to S3
            img_buffer = io.BytesIO(img_bytes)
            object_name = f"facetoon/{book_uuid}/page_0.png"
            
            upload_result = upload_file_object_to_s3(img_buffer, object_name=object_name)
            
            if not upload_result['success']:
                raise ValueError(f"Failed to upload cover page: {upload_result['message']}")
            
            print(f"[Cover] ✓ Generated cover page as page 0")
            
            return {
                'image_urls': {
                    'page 0': upload_result['url']
                },
                'full_image_urls': {
                    'page 0': upload_result['url']
                },
                'image_bytes': {
                    'page 0': img_bytes
                }
            }
            
        except Exception as e:
            print(f"[Cover] Error generating cover page: {str(e)}")
            raise
    
    async def _swap_character_in_page(
        self,
        page_key: str,
        page_url: str,
        prompt: str,
        story_text: str,
        character_name: str,
        gender: str,
        age: int,
        image_style: str,
        reference_images_bytes: List[bytes],
        book_uuid: str,
        page_number: int,
        font_size: int = 100,
        text_color: str = "white",
        dpi: int = 300
    ) -> Dict:
        """Swap character in a single page using existing image as composition reference"""
        try:
            print(f"\n[Swap] Processing {page_key}...")
            
            # Download existing image
            print(f"[Swap] Downloading existing image: {page_url}")
            existing_img_response = requests.get(page_url, timeout=60)
            existing_img_response.raise_for_status()
            existing_img_bytes = existing_img_response.content
            
            # Replace old character name with new one in prompt
            enhanced_prompt = self._create_swap_prompt(
                prompt, story_text, character_name, gender, age, image_style
            )
            
            # Generate new image with character swap
            new_img_bytes = await asyncio.to_thread(
                self._generate_swapped_image,
                enhanced_prompt,
                reference_images_bytes,
                existing_img_bytes,
                gender,
                age,
                image_style
            )
            
            # Add text to image
            img = Image.open(io.BytesIO(new_img_bytes))
            
            if story_text and story_text.strip():
                print(f"[Swap] Adding text to {page_key}...")
                try:
                    text_recommendation = await get_text_placement_recommendation(
                        new_img_bytes, story_text, font_size
                    )
                    
                    draw = ImageDraw.Draw(img)
                    
                    # Load font
                    from pathlib import Path
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
                    
                    # Update image bytes
                    output_buffer = io.BytesIO()
                    img.save(output_buffer, format='PNG', dpi=(dpi, dpi))
                    output_buffer.seek(0)
                    new_img_bytes = output_buffer.read()
                    img = Image.open(io.BytesIO(new_img_bytes))
                    
                    print(f"[Swap] Text added successfully")
                except Exception as e:
                    print(f"[Swap] Error adding text: {str(e)}")
            
            # Upload full image
            page_num_str = page_key.replace('page ', '').replace(' ', '_')
            full_object_name = f"facetoon/{book_uuid}/full/image_{page_num_str}.png"
            
            full_img_buffer = io.BytesIO()
            img.save(full_img_buffer, format='PNG', dpi=(dpi, dpi))
            full_img_buffer.seek(0)
            full_img_bytes_data = full_img_buffer.read()
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
            
            print(f"[Swap] ✓ Completed {page_key} -> pages {left_page_num}, {right_page_num}")
            
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
            print(f"[Swap] Error processing {page_key}: {str(e)}")
            raise
    
    async def _generate_coloring_page(
        self,
        page_key: str,
        prompt: str,
        character_name: str,
        gender: str,
        age: int,
        image_style: str,
        reference_images_bytes: List[bytes],
        book_uuid: str
    ) -> Dict:
        """Generate coloring page from scratch (no existing image)"""
        try:
            print(f"\n[Coloring] Generating {page_key}...")
            
            # Replace character name in prompt
            enhanced_prompt = f"""
Children's storybook coloring page in {image_style} style.
Main character: {age}-year-old {gender} child named {character_name} matching the reference image exactly.
{prompt}
Black and white line art, suitable for coloring. Clean outlines, no colors, no shading.
The child's face, features, hair, and appearance must exactly match the reference image provided.
ABSOLUTELY NO TEXT, LETTERS, WORDS, OR ANY WRITTEN CHARACTERS IN THE IMAGE.
""".strip()
            
            # Generate image
            img_bytes = await asyncio.to_thread(
                self._generate_single_image_from_scratch,
                enhanced_prompt,
                reference_images_bytes,
                2550, 2550  # Square format for coloring pages
            )
            
            # Upload to S3
            page_num = int(page_key.split()[1])
            final_page_num = 23 if page_num == 12 else 24
            
            img_buffer = io.BytesIO(img_bytes)
            object_name = f"facetoon/{book_uuid}/page_{final_page_num}.png"
            
            upload_result = upload_file_object_to_s3(img_buffer, object_name=object_name)
            
            if not upload_result['success']:
                raise ValueError(f"Failed to upload coloring page: {upload_result['message']}")
            
            print(f"[Coloring] ✓ Generated {page_key} as page {final_page_num}")
            
            return {
                'image_urls': {
                    f'page {final_page_num}': upload_result['url']
                },
                'full_image_urls': {
                    page_key: upload_result['url']
                },
                'image_bytes': {
                    f'page {final_page_num}': img_bytes
                }
            }
            
        except Exception as e:
            print(f"[Coloring] Error generating {page_key}: {str(e)}")
            raise
    
    def _generate_swapped_image(
        self,
        prompt: str,
        reference_images_bytes: List[bytes],
        existing_image_bytes: bytes,
        gender: str,
        age: int,
        image_style: str,
        width: int = 5100,
        height: int = 2550
    ) -> bytes:
        """Generate image with swapped character using SeeDream API"""
        try:
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Encode images to base64
            reference_image_base64 = base64.b64encode(reference_images_bytes[0]).decode('utf-8')
            existing_image_base64 = base64.b64encode(existing_image_bytes).decode('utf-8')
            
            # Payload: use reference images for character, existing image for composition/style
            payload = {
                'model': self.model,
                'prompt': prompt,
                'width': width,
                'height': height,
                'watermark': False,
                'reference_image': reference_image_base64,  # New character face
                'style_reference_image': existing_image_base64  # Composition reference
            }
            
            # Add additional reference images if available
            if len(reference_images_bytes) > 1:
                for idx, img_bytes in enumerate(reference_images_bytes[1:3], 2):
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    payload[f'reference_image_{idx}'] = img_base64
            
            # Call SeeDream API
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract image URL
            image_url = result.get('data', [{}])[0].get('url') or result.get('image_url') or result.get('url')
            
            if not image_url:
                raise ValueError("No image URL in SeeDream response")
            
            # Download and resize
            img_response = requests.get(image_url, timeout=60)
            img_response.raise_for_status()
            
            img = Image.open(io.BytesIO(img_response.content))
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Convert to bytes
            output_buffer = io.BytesIO()
            img.save(output_buffer, format='PNG', dpi=(300, 300))
            output_buffer.seek(0)
            
            return output_buffer.read()
            
        except Exception as e:
            print(f"Error calling SeeDream API for swap: {str(e)}")
            raise
    
    def _generate_single_image_from_scratch(
        self,
        prompt: str,
        reference_images_bytes: List[bytes],
        width: int,
        height: int
    ) -> bytes:
        """Generate image from scratch (for coloring pages)"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.model,
                'prompt': prompt,
                'width': width,
                'height': height,
                'watermark': False
            }
            
            # Add reference images
            if reference_images_bytes:
                for idx, img_bytes in enumerate(reference_images_bytes[:3], 1):
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    if idx == 1:
                        payload['reference_image'] = img_base64
                    else:
                        payload[f'reference_image_{idx}'] = img_base64
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            image_url = result.get('data', [{}])[0].get('url') or result.get('image_url') or result.get('url')
            
            if not image_url:
                raise ValueError("No image URL in response")
            
            img_response = requests.get(image_url, timeout=60)
            img_response.raise_for_status()
            
            img = Image.open(io.BytesIO(img_response.content))
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            output_buffer = io.BytesIO()
            img.save(output_buffer, format='PNG', dpi=(300, 300))
            output_buffer.seek(0)
            
            return output_buffer.read()
            
        except Exception as e:
            print(f"Error generating image from scratch: {str(e)}")
            raise
    
    def _create_swap_prompt(
        self,
        original_prompt: str,
        story: str,
        character_name: str,
        gender: str,
        age: int,
        image_style: str
    ) -> str:
        """Create enhanced prompt for character swap"""
        gender_desc = "boy" if gender.lower() == "male" else "girl"
        
        # Replace any character names in the original prompt with new name
        # This is a simple replacement - you might want to make this more sophisticated
        enhanced_prompt = f"""
Children's storybook illustration in {image_style} style.
Main character: {age}-year-old {gender_desc} named {character_name} matching the reference image exactly.
{original_prompt}
Maintain the exact same scene composition, setting, background, and overall layout from the style reference.
CRITICAL: Only change the character's appearance to match the new reference images - same facial features, hair, skin tone as reference.
Keep everything else identical: scene, actions, objects, colors, lighting, perspective.
ABSOLUTELY NO TEXT, LETTERS, WORDS, SIGNS, LABELS, OR ANY WRITTEN CHARACTERS IN THE IMAGE.
""".strip()
        
        return enhanced_prompt
    
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
                    
                    # Split pairs (1-2, 3-4, etc.)
                    if page_num % 2 == 1 and page_num > 0 and page_num < 23:
                        right_page_num = page_num + 1
                        right_key = f'page {right_page_num}'
                        
                        if right_key in image_urls:
                            original_page_num = (page_num + 1) // 2
                            original_page_key = f'page {original_page_num}'
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
                            page_obj = PageImageUrls(
                                name=key,
                                fullPageUrl=url,
                                leftUrl=None,
                                rightUrl=None
                            )
                            structured_pages.append(page_obj)
                            processed_pages.add(page_num)
                    elif page_num == 0 or page_num >= 23:
                        if page_num not in processed_pages:
                            page_obj = PageImageUrls(
                                name=key,
                                fullPageUrl=url,
                                leftUrl=None,
                                rightUrl=None
                            )
                            structured_pages.append(page_obj)
                            processed_pages.add(page_num)
                    elif page_num % 2 == 0:
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



