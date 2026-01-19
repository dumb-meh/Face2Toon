import io
import os
import asyncio
import traceback
from pathlib import Path
from typing import Dict
from PIL import Image, ImageDraw, ImageFont
from app.utils.image_analysis import get_text_placement_recommendation
from app.utils.upload_to_bucket import upload_file_object_to_s3


async def text_insertion_worker(
    text_queue: asyncio.Queue,
    results_dict: Dict,
    story: Dict[str, str],
    font_size: int,
    text_color: str,
    dpi: int,
    session_id: str,
    upload_to_s3: bool,
    book_uuid: str,
    language: str = "English"
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
            print(f"[Text Worker] - is_single_page: {is_single_page}")
            
            # Get story text for this page
            story_text = story.get(page_key, "")
            print(f"[Text Worker] - story_text length: {len(story_text) if story_text else 0}")
            
            # First, save the original image WITHOUT text as the full image
            img_original = Image.open(io.BytesIO(image_bytes))
            print(f"[Text Worker] Opened original image: {img_original.size[0]}x{img_original.size[1]} pixels")
            
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
                        full_image_url = upload_result['url']
                        print(f"[Text Worker] Uploaded coloring page to S3 as page_{final_page_num}: {full_image_url}")
                    else:
                        full_image_url = None
                        print(f"[Text Worker] Failed to upload coloring page to S3")
                else:
                    os.makedirs('uploads/generated_images/splitted', exist_ok=True)
                    full_image_filename = f"uploads/generated_images/splitted/{session_id}_page_{final_page_num}.png"
                    img_original.save(full_image_filename, format='PNG', dpi=(dpi, dpi))
                    base_url = os.getenv('domain') or os.getenv('BASE_URL')
                    base_url = base_url.rstrip('/')
                    full_image_url = f"{base_url}/{full_image_filename}"
                    print(f"[Text Worker] Saved coloring page locally as page_{final_page_num}: {full_image_url}")
            else:
                # Upload/save full image WITHOUT text (for future reuse in swap_story_information)
                if upload_to_s3:
                    full_object_name = f"facetoon/{book_uuid}/full/image_{page_num_str}.png"
                    upload_result = upload_file_object_to_s3(full_image_bytes_no_text, object_name=full_object_name)
                    if upload_result['success']:
                        full_image_url = upload_result['url']
                        print(f"[Text Worker] Uploaded full image WITHOUT text to S3: {full_image_url}")
                    else:
                        full_image_url = None
                        print(f"[Text Worker] Failed to upload full image to S3")
                else:
                    os.makedirs('uploads/generated_images', exist_ok=True)
                    full_image_filename = f"uploads/generated_images/{session_id}_image_{page_num_str}.png"
                    img_original.save(full_image_filename, format='PNG', dpi=(dpi, dpi))
                    base_url = os.getenv('domain') or os.getenv('BASE_URL')
                    base_url = base_url.rstrip('/')
                    full_image_url = f"{base_url}/{full_image_filename}"
                    print(f"[Text Worker] Saved full image WITHOUT text locally: {full_image_url}")
            
            # Now work with a copy of the image to add text (if needed)
            img_with_text = img_original.copy()
            
            # Add text if needed
            text_was_added = False
            if story_text and not is_single_page:
                # Use image analysis to add text
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
                        font_path = Path(__file__).resolve().parents[2] / "fonts" / "Playpen_Sans_Arabic" / "PlaypenSansArabic-Regular.ttf"
                    else:
                        font_path = Path(__file__).resolve().parents[2] / "fonts" / "Comic_Relief" / "ComicRelief-Regular.ttf"
                    
                    try:
                        if font_path.exists():
                            font = ImageFont.truetype(str(font_path), font_size)
                        else:
                            print(f"[Text Worker] Warning: Font not found at {font_path}, using default font")
                            font = ImageFont.load_default()
                    except Exception as font_err:
                        print(f"[Text Worker] Warning: Error loading font: {font_err}, using default font")
                        font = ImageFont.load_default()
                    
                    # Draw text with outline
                    outline_color = "black" if text_color.lower() == "white" else "white"
                    
                    for line_coord in text_recommendation.line_coordinates:
                        x, y = line_coord.x, line_coord.y
                        line_text = line_coord.text
                        
                        # Draw outline
                        for adj_x in range(-2, 3):
                            for adj_y in range(-2, 3):
                                draw.text((x + adj_x, y + adj_y), line_text, font=font, fill=outline_color)
                        
                        # Draw main text
                        draw.text((x, y), line_text, font=font, fill=text_color)
                    
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
                middle_x = img_with_text.width // 2
                left_half = img_with_text.crop((0, 0, middle_x, img_with_text.height))
                right_half = img_with_text.crop((middle_x, 0, img_with_text.width, img_with_text.height))
                
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
                        print(f"[Text Worker] ✓ Completed {page_key} -> pages {left_page_num}, {right_page_num}")
                        print(f"[Text Worker]   Full (no text): {full_image_url}")
                        print(f"[Text Worker]   Left (with text): {left_result['url']}")
                        print(f"[Text Worker]   Right (with text): {right_result['url']}")
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
                    print(f"[Text Worker] ✓ Completed {page_key} -> pages {left_page_num}, {right_page_num}")
                    print(f"[Text Worker]   Full (no text): {full_image_url}")
                    print(f"[Text Worker]   Left (with text): {left_filename}")
                    print(f"[Text Worker]   Right (with text): {right_filename}")
            else:
                # Single page (cover, coloring pages, or back cover) - use original without text
                page_num = int(page_key.split()[1]) if page_key.startswith('page ') and page_key.split()[1].isdigit() else None
                if page_num == 12:
                    return_key = 'page 23'
                elif page_num == 13:
                    return_key = 'page 24'
                elif page_key == 'page last page':
                    return_key = 'page 25'
                else:
                    return_key = page_key
                
                results_dict['image_urls'][return_key] = full_image_url
                results_dict['full_image_urls'][page_key] = full_image_url
                results_dict['image_bytes'][return_key] = full_image_bytes_no_text_data
                print(f"[Text Worker] ✓ Completed single page {page_key} as {return_key}")
                print(f"[Text Worker]   URL: {full_image_url}")
            
            text_queue.task_done()
            
        except Exception as e:
            print(f"[Text Worker] Error processing item: {str(e)}")
            traceback.print_exc()
            text_queue.task_done()
