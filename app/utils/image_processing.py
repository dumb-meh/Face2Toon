import io
import requests
from PIL import Image
from app.services.Generate_Images.generate_images_schema import PageImageUrls


def resize_image_to_print_size(image_url: str, page_key: str) -> tuple[bytes, bool]:
    """Download image and resize to exact physical dimensions
    Pages 0, 12, 13, 14 (last page): 8.5" x 8.5" at 300 DPI (cover, coloring pages, and back cover)
    Other pages: 17" width x 8.5" height at 300 DPI
    Returns: (image_bytes, is_single_page)"""
    try:
        # Download the image from Seedream
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        
        # Open image with PIL
        img = Image.open(io.BytesIO(response.content))
        
        # Determine if this is a single page (cover, coloring pages, or back cover)
        if page_key == 'page last page':
            page_num = 14
        elif page_key.startswith('page ') and page_key.split()[1].isdigit():
            page_num = int(page_key.split()[1])
        else:
            page_num = 0
        
        is_single_page = page_num == 0 or page_num == 12 or page_num == 13 or page_num == 14
        
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
    
    structured_pages = []
    processed_pages = set()
    
    print(f"\n[Convert] Converting URLs to structured format")
    print(f"[Convert] image_urls keys: {sorted(image_urls.keys())}")
    print(f"[Convert] full_image_urls keys: {sorted(full_image_urls.keys())}")
    
    # Helper function to get sort key for page names
    def get_sort_key(key):
        if 'page' in key:
            parts = key.split()
            if len(parts) > 1:
                if parts[1] == 'last':
                    return 999  # Sort "page last page" at the end
                elif parts[1].isdigit():
                    return int(parts[1])
        return 0
    
    # Sort keys by page number
    sorted_keys = sorted(image_urls.keys(), key=get_sort_key)
    
    for key in sorted_keys:
        url = image_urls[key]
        
        # Handle "page last page" as page 14 (back cover)
        if key == 'page last page':
            page_obj = PageImageUrls(
                name="page 14",
                fullPageUrl=full_image_urls.get(key, url),
                leftUrl=None,
                rightUrl=None
            )
            structured_pages.append(page_obj)
            continue
        
        if 'page' in key:
            page_num_str = key.split()[1]
            if page_num_str.isdigit():
                page_num = int(page_num_str)
                
                if page_num in processed_pages:
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
                
                # Coloring pages (stored as page 23, page 24 to avoid conflict with splits)
                # Return them as page 12 and page 13 in the API response
                if page_num == 23:
                    page_obj = PageImageUrls(
                        name="page 12",
                        fullPageUrl=full_image_urls.get(key, url),
                        leftUrl=None,
                        rightUrl=None
                    )
                    structured_pages.append(page_obj)
                    processed_pages.add(page_num)
                    continue
                
                if page_num == 24:
                    page_obj = PageImageUrls(
                        name="page 13",
                        fullPageUrl=full_image_urls.get(key, url),
                        leftUrl=None,
                        rightUrl=None
                    )
                    structured_pages.append(page_obj)
                    processed_pages.add(page_num)
                    continue
                
                # Regular split pairs (1-2, 3-4, etc.)
                if page_num % 2 == 1 and page_num > 0:
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
                
                # Even numbered pages that weren't paired (shouldn't happen normally)
                if page_num % 2 == 0 and page_num not in processed_pages:
                    page_obj = PageImageUrls(name=key, fullPageUrl=url, leftUrl=None, rightUrl=None)
                    structured_pages.append(page_obj)
                    processed_pages.add(page_num)
    
    print(f"[Convert] Created {len(structured_pages)} structured page objects")
    return structured_pages
