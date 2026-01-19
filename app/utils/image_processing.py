import io
import requests
from PIL import Image


def resize_image_to_print_size(image_url: str, page_key: str, session_id: str) -> tuple[bytes, bool]:
    """Download image and resize to exact physical dimensions
    Pages 0, 12, 13, 'page last page': 8.5" x 8.5" at 300 DPI (cover, coloring pages, and back cover)
    Other pages: 17" width x 8.5" height at 300 DPI
    Returns: (image_bytes, is_single_page)"""
    try:
        # Download the image from Seedream
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        
        # Open image with PIL
        img = Image.open(io.BytesIO(response.content))
        
        # Determine if this is a single page (cover, coloring pages, or back cover)
        page_num = int(page_key.split()[1]) if page_key.startswith('page ') and page_key.split()[1].isdigit() else None
        is_single_page = page_key == 'page 0' or page_num == 12 or page_num == 13 or page_key == 'page last page'
        
        # Physical dimensions in inches
        dpi = 300
        
        if is_single_page:
            # Square pages: cover, coloring pages, and back cover
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


def convert_dict_to_structured(image_urls: dict, full_image_urls: dict):
    """Convert flat dictionary to structured format with PageImageUrls objects"""
    from .generate_images_schema import PageImageUrls
    
    structured_pages = []
    processed_pages = set()
    
    print(f"\n[Convert] Converting URLs to structured format")
    print(f"[Convert] image_urls keys: {sorted(image_urls.keys())}")
    print(f"[Convert] full_image_urls keys: {sorted(full_image_urls.keys())}")
    
    # Sort keys by page number
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
                            name="page 12",
                            fullPageUrl=None,
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
                        # Calculate original page number (reverse split calculation)
                        original_page = (page_num - 1) // 2 + 1
                        original_page_key = f"page {original_page}"
                        full_url = full_image_urls.get(original_page_key)
                        
                        page_obj = PageImageUrls(
                            name=original_page_key,
                            fullPageUrl=full_url,
                            leftUrl=url,
                            rightUrl=image_urls[right_key]
                        )
                        structured_pages.append(page_obj)
                        processed_pages.add(page_num)
                        processed_pages.add(right_page_num)
                        continue
                
            page_num_str = key.split()[1]
            if page_num_str.isdigit():
                page_num = int(page_num_str)
                
                # Skip if already processed
                if page_num in processed_pages:
                    continue
                
                # Check if this is part of a split pair (odd pages 1, 3, 5, etc. pair with even pages 2, 4, 6, etc.)
                # Exception: pages 23 and 24 are separate single pages (coloring pages), not a split pair
                if page_num % 2 == 1 and page_num > 0 and page_num < 23:
                    right_page_num = page_num + 1
                    right_key = f'page {right_page_num}'
                    
                    if right_key in image_urls:
                        # Calculate original page number (reverse split calculation)
                        original_page = (page_num - 1) // 2 + 1
                        original_page_key = f"page {original_page}"
                        full_url = full_image_urls.get(original_page_key)
                        
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
                        # Odd page without pair
                        page_obj = PageImageUrls(name=key, fullPageUrl=url, leftUrl=None, rightUrl=None)
                        structured_pages.append(page_obj)
                        processed_pages.add(page_num)
                elif page_num == 0 or page_num >= 23:
                    # Cover page (0) or coloring pages (23, 24)
                    if page_num not in processed_pages:
                        page_obj = PageImageUrls(name=key, fullPageUrl=url, leftUrl=None, rightUrl=None)
                        structured_pages.append(page_obj)
                        processed_pages.add(page_num)
                elif page_num % 2 == 0:
                    # Even numbered pages that weren't paired (shouldn't happen normally)
                    if page_num not in processed_pages:
                        page_obj = PageImageUrls(name=key, fullPageUrl=url, leftUrl=None, rightUrl=None)
                        structured_pages.append(page_obj)
                        processed_pages.add(page_num)
    
    print(f"[Convert] Created {len(structured_pages)} structured page objects")
    return structured_pages
