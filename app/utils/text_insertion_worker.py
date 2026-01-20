import io
import os
import asyncio
import traceback
from io import BytesIO
from pathlib import Path
from typing import Dict
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from app.utils.image_analysis import get_text_placement_recommendation
from app.utils.upload_to_bucket import upload_file_object_to_s3


async def process_single_page_text(
    page_key: str,
    image_bytes: bytes,
    page_number: int,
    is_single_page: bool,
    story: Dict[str, str],
    font_size: int,
    text_color: str,
    dpi: int,
    session_id: str,
    upload_to_s3: bool,
    book_uuid: str,
    language: str = "English"
):
    """Process text insertion for a single page (replaces worker queue system)"""
    try:
        print(f"[Text Processor] Processing {page_key}...")
        
        # Get story text for this page
        story_text = story.get(page_key, "")
        
        # First, save the original image WITHOUT text as the full image
        img_original = Image.open(io.BytesIO(image_bytes))
        print(f"[Text Processor] Opened image: {img_original.size[0]}x{img_original.size[1]} pixels")
        
        page_num_str = page_key.replace('page ', '').replace(' ', '_')
        
        # Save full image bytes WITHOUT text
        full_image_bytes_no_text = io.BytesIO()
        img_original.save(full_image_bytes_no_text, format='PNG', dpi=(dpi, dpi))
        full_image_bytes_no_text.seek(0)
        full_image_bytes_no_text_data = full_image_bytes_no_text.read()
        full_image_bytes_no_text.seek(0)
        
        # Determine page number for special pages
        if page_key == 'page last page':
            page_num = 14  # Back cover
        elif page_key.startswith('page ') and page_key.split()[1].isdigit():
            page_num = int(page_key.split()[1])
        else:
            page_num = 0
        
        # For coloring pages (12, 13), upload to splitted/ with page_23/page_24 naming
        # For other single pages, upload to full/
        if page_num == 12 or page_num == 13:
            # Coloring pages go to splitted folder
            final_page_num = 23 if page_num == 12 else 24
            if upload_to_s3:
                full_object_name = f"facetoon/{book_uuid}/splitted/page_{final_page_num}.png"
                upload_result = upload_file_object_to_s3(full_image_bytes_no_text, object_name=full_object_name)
                if upload_result['success']:
                    full_page_url = upload_result['url']
                    print(f"[Text Worker] Uploaded coloring page to S3 as page_{final_page_num}: {full_page_url}")
                else:
                    # Fallback to placeholder if upload fails
                    full_page_url = f"s3://mycvconnect/facetoon/{book_uuid}/splitted/page_{final_page_num}.png"
                    print(f"[Text Worker] Failed to upload coloring page to S3, using placeholder: {full_page_url}")
            else:
                os.makedirs('uploads/generated_images/splitted', exist_ok=True)
                full_image_filename = f"uploads/generated_images/splitted/{session_id}_page_{final_page_num}.png"
                img_original.save(full_image_filename, format='PNG', dpi=(dpi, dpi))
                base_url = os.getenv('domain') or os.getenv('BASE_URL')
                base_url = base_url.rstrip('/')
                full_page_url = f"{base_url}/{full_image_filename}"
                print(f"[Text Worker] Saved coloring page locally as page_{final_page_num}: {full_page_url}")
        else:
            # Upload/save full image WITHOUT text (for future reuse in swap_story_information)
            if upload_to_s3:
                full_object_name = f"facetoon/{book_uuid}/full/image_{page_num_str}.png"
                upload_result = upload_file_object_to_s3(full_image_bytes_no_text, object_name=full_object_name)
                if upload_result['success']:
                    full_page_url = upload_result['url']
                    print(f"[Text Worker] Uploaded full image WITHOUT text to S3: {full_page_url}")
                else:
                    # Fallback to placeholder if upload fails
                    full_page_url = f"s3://mycvconnect/facetoon/{book_uuid}/full/image_{page_num_str}.png"
                    print(f"[Text Worker] Failed to upload full image to S3, using placeholder: {full_page_url}")
            else:
                os.makedirs('uploads/generated_images', exist_ok=True)
                full_image_filename = f"uploads/generated_images/{session_id}_image_{page_num_str}.png"
                img_original.save(full_image_filename, format='PNG', dpi=(dpi, dpi))
                base_url = os.getenv('domain') or os.getenv('BASE_URL')
                base_url = base_url.rstrip('/')
                full_page_url = f"{base_url}/{full_image_filename}"
                print(f"[Text Worker] Saved full image WITHOUT text locally: {full_page_url}")
        
        # Now work with a copy of the image to add text (if needed)
        # IMPORTANT: img_with_text is a COPY, so img_original remains WITHOUT text
        # The full image (img_original) was already saved/uploaded WITHOUT text above
        img_with_text = img_original.copy()
        
        # Add text if needed
        text_was_added = False
        if story_text and not is_single_page:
            # Use image analysis to add text TO THE COPY (not the original)
            try:
                print(f"[Text Worker] Adding text to {page_key}...")
                
                # Get text placement recommendation
                text_recommendation = await get_text_placement_recommendation(
                    image_bytes, 
                    story_text, 
                    font_size
                )
                
                print(f"[Text Worker] Got recommendation: {text_recommendation.number_of_lines} lines on {text_recommendation.side} side")
                
                # Draw on the copy
                draw = ImageDraw.Draw(img_with_text)
                
                # Load font based on language
                if language and language.lower() == "arabic":
                    font_path = Path(__file__).resolve().parents[2] / "fonts" / "Playpen_Sans_Arabic" / "PlaypenSansArabic-SemiBold.ttf"
                else:
                    font_path = Path(__file__).resolve().parents[2] / "fonts" / "Comic_Relief" / "ComicRelief-Bold.ttf"
                
                try:
                    if font_path.exists():
                        font = ImageFont.truetype(str(font_path), font_size)
                    else:
                        print(f"[Text Worker] Warning: Font not found at {font_path}, using default font")
                        font = ImageFont.load_default()
                except Exception as font_err:
                    print(f"[Text Worker] Warning: Error loading font: {font_err}, using default font")
                    font = ImageFont.load_default()
                
                # Apply purple glow effect
                width, height = img_with_text.size
                
                # Step 1: Create glow mask - draw text in white on black background
                glow_mask = Image.new('L', (width, height), 0)
                mask_draw = ImageDraw.Draw(glow_mask)
                
                # Draw all text lines on the mask
                for line_coord in text_recommendation.line_coordinates:
                    mask_draw.text((line_coord.x, line_coord.y), line_coord.text, font=font, fill=255)
                
                # Step 2: Apply multiple Gaussian blurs for strong purple glow effect
                glow1 = glow_mask.filter(ImageFilter.GaussianBlur(radius=40))  # Wide glow
                glow2 = glow_mask.filter(ImageFilter.GaussianBlur(radius=25))  # Medium glow
                glow3 = glow_mask.filter(ImageFilter.GaussianBlur(radius=15))  # Tight glow
                
                # Step 3: Create bright purple glow color
                purple_glow = Image.new('RGB', (width, height), (180, 60, 200))
                
                # Step 4: Apply each glow layer to the image (builds up intensity)
                img_with_text.paste(purple_glow, (0, 0), glow1)  # Widest glow
                img_with_text.paste(purple_glow, (0, 0), glow2)  # Medium glow (makes it stronger)
                img_with_text.paste(purple_glow, (0, 0), glow3)  # Tight glow (adds more intensity)
                
                # Step 5: Draw white text on top of the purple glow
                draw = ImageDraw.Draw(img_with_text)
                text_color_final = (255, 255, 255)
                
                for line_coord in text_recommendation.line_coordinates:
                    draw.text((line_coord.x, line_coord.y), line_coord.text, font=font, fill=text_color_final)
                
                text_was_added = True
                print(f"[Text Worker] ✓ Successfully added {len(text_recommendation.line_coordinates)} lines of text to {page_key}")
                
            except Exception as e:
                print(f"[Text Worker] ✗ Error adding text to {page_key}: {str(e)}")
                import traceback
                traceback.print_exc()
                text_was_added = False
        else:
            if not story_text:
                print(f"[Text Worker] No story text for {page_key}, skipping text insertion")
            if is_single_page:
                print(f"[Text Worker] {page_key} is single page (cover/coloring), skipping text insertion")
        
        # Split if not single page - use the image WITH text for left/right pages
        if not is_single_page:
            try:
                print(f"[Text Worker] Splitting {page_key}...")
                width, height = img_with_text.size
                
                # Split into left and right halves
                left_img = img_with_text.crop((0, 0, width // 2, height))
                right_img = img_with_text.crop((width // 2, 0, width, height))
                
                # Save to bytes
                left_bytes = BytesIO()
                right_bytes = BytesIO()
                left_img.save(left_bytes, format='PNG')
                right_img.save(right_bytes, format='PNG')
                left_bytes.seek(0)
                right_bytes.seek(0)
                
                # Calculate left and right page numbers
                left_page_num = (page_number * 2) + 1
                right_page_num = (page_number * 2) + 2
                
                # Read bytes for return before uploading
                left_bytes_data = left_bytes.read()
                right_bytes_data = right_bytes.read()
                left_bytes.seek(0)
                right_bytes.seek(0)
                
                # Upload splits with page_X.png naming
                left_object_name = f"facetoon/{book_uuid}/splitted/page_{left_page_num}.png"
                right_object_name = f"facetoon/{book_uuid}/splitted/page_{right_page_num}.png"
                
                left_result = upload_file_object_to_s3(left_bytes, object_name=left_object_name)
                right_result = upload_file_object_to_s3(right_bytes, object_name=right_object_name)
                
                left_url = left_result.get('url') if left_result.get('success') else None
                right_url = right_result.get('url') if right_result.get('success') else None
                
                print(f"[Text Worker] ✓ Split {page_key} -> page {left_page_num} (left), page {right_page_num} (right)")
                
                # Return structure for split pages with page numbers and bytes
                return {
                    "pageKey": page_key,
                    "leftPageUrl": left_url or "placeholder_left",
                    "rightPageUrl": right_url or "placeholder_right",
                    "leftPageNum": left_page_num,
                    "rightPageNum": right_page_num,
                    "leftPageBytes": left_bytes_data,
                    "rightPageBytes": right_bytes_data,
                    "fullPageUrl": full_page_url,
                    "wasProcessed": text_was_added,
                    "pageNumber": page_number
                }
                
            except Exception as split_err:
                print(f"[Text Worker] ✗ Error splitting {page_key}: {str(split_err)}")
                import traceback
                traceback.print_exc()
                left_page_num = (page_number * 2) + 1
                right_page_num = (page_number * 2) + 2
                return {
                    "pageKey": page_key,
                    "leftPageUrl": "error_splitting_left",
                    "rightPageUrl": "error_splitting_right",
                    "leftPageNum": left_page_num,
                    "rightPageNum": right_page_num,
                    "fullPageUrl": full_page_url,
                    "wasProcessed": False,
                    "pageNumber": page_number
                }
        else:
            # Single page - return with proper structure
            print(f"[Text Worker] ✓ Single page {page_key} completed")
            
            # For single pages, we need to get the image bytes for PDF
            # Read the image with text for PDF
            single_page_bytes = BytesIO()
            img_with_text.save(single_page_bytes, format='PNG', dpi=(dpi, dpi))
            single_page_bytes.seek(0)
            single_page_bytes_data = single_page_bytes.read()
            
            # For coloring pages, use the remapped page number (23, 24) to avoid conflict
            return_page_key = page_key
            if page_num == 12:
                return_page_key = "page 23"
            elif page_num == 13:
                return_page_key = "page 24"
            
            return {
                "pageKey": return_page_key,
                "leftPageUrl": None,
                "rightPageUrl": None,
                "fullPageUrl": full_page_url,
                "singlePageBytes": single_page_bytes_data,
                "wasProcessed": text_was_added,
                "pageNumber": page_number
            }
            
    except Exception as e:
        print(f"[Text Worker] ✗ Error processing {page_key}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
