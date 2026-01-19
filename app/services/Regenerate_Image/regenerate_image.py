import os
import io
import re
import requests
import base64
import boto3
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
from .regenerate_image_schema import ReGenerateImageRequest, ReGenerateImageResponse, PageImageUrls
from app.utils.upload_to_bucket import upload_file_object_to_s3, delete_file_from_s3
from app.utils.image_analysis import get_text_placement_recommendation

load_dotenv()

class ReGenerateImage:
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
    
    def regenerate_image(self, request: ReGenerateImageRequest) -> ReGenerateImageResponse:
        """Main method to regenerate an image with error corrections"""
        try:
            # Extract the S3 object key from the image URL
            s3_object_key = self._extract_s3_key_from_url(request.iamge_url)
            
            # Download the image from S3
            print(f"Downloading image from S3: {s3_object_key}")
            image_bytes = self._download_image_from_s3(s3_object_key)
            
            is_cover = request.page_type.lower() == "cover"
            
            if is_cover:
                # Cover page: only delete full image, no splitting
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit deletion task for full image only
                    delete_full_future = executor.submit(self._delete_old_image, s3_object_key)
                    
                    # Generate cover image (no splitting)
                    image_urls_future = executor.submit(
                        self._generate_cover_image,
                        request.prompt,
                        request.story,
                        image_bytes,
                        request.gender,
                        request.age,
                        request.image_style,
                        s3_object_key
                    )
                    
                    # Wait for both tasks
                    delete_full_future.result()
                    page_image_urls = image_urls_future.result()
            else:
                # Normal page: delete all three images and split
                left_key, right_key = self._derive_split_image_keys(s3_object_key)
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    # Submit deletion tasks for full and split images
                    delete_full_future = executor.submit(self._delete_old_image, s3_object_key)
                    delete_left_future = executor.submit(self._delete_old_image, left_key)
                    delete_right_future = executor.submit(self._delete_old_image, right_key)
                    
                    # Generate new image with splitting
                    image_urls_future = executor.submit(
                        self._generate_and_split_image,
                        request.prompt,
                        request.story,
                        image_bytes,
                        request.gender,
                        request.age,
                        request.image_style,
                        s3_object_key,
                        left_key,
                        right_key,
                        request.page_number
                    )
                    
                    # Wait for all tasks
                    delete_full_future.result()
                    delete_left_future.result()
                    delete_right_future.result()
                    page_image_urls = image_urls_future.result()
            
            print(f"Image regenerated successfully: {page_image_urls.fullPageUrl}")
            
            # Update the PDF with the regenerated page(s)
            print(f"Updating PDF with regenerated page(s)...")
            updated_pdf_url = self._update_pdf_with_new_pages(
                request.pdf_url,
                page_image_urls,
                request.page_number,
                is_cover
            )
            
            return ReGenerateImageResponse(
                image_url=[page_image_urls],
                pdf_url=updated_pdf_url
            )
            
        except Exception as e:
            print(f"Error in regenerate_image: {str(e)}")
            raise
    
    def _extract_s3_key_from_url(self, url: str) -> str:
        """Extract S3 object key from full URL"""
        # Parse URL like: https://mycvconnect.s3.eu-north-1.amazonaws.com/facetoon/book_uuid/image.png
        # Or local URL: http://localhost:8000/uploads/generated_images/20260101_071108_83c28c78_image_1.png
        if 's3.' in url or '.amazonaws.com' in url:
            parsed = urlparse(url)
            s3_key = parsed.path.lstrip('/')
        else:
            # Local URL pattern - extract the path after /uploads/
            parsed = urlparse(url)
            path = parsed.path.lstrip('/')
            # Remove 'uploads/' prefix if present
            if path.startswith('uploads/'):
                s3_key = path[len('uploads/'):]
            else:
                s3_key = path
        return s3_key
    
    def _derive_split_image_keys(self, full_image_key: str) -> tuple:
        """Derive the split image S3 keys from the full image key
        Example: 
        Full: generated_images/20260101_071108_83c28c78_image_1.png
        Left: generated_images/splitted/20260101_071108_83c28c78_page_1.png
        Right: generated_images/splitted/20260101_071108_83c28c78_page_2.png
        
        Or:
        Full: facetoon/uuid/full/image_2.png
        Left: facetoon/uuid/splitted/page_5.png
        Right: facetoon/uuid/splitted/page_6.png
        """
        # Extract the directory and filename
        directory = os.path.dirname(full_image_key)
        filename = os.path.basename(full_image_key)
        
        # Try pattern 1: {session_id}_image_{number}.png
        match = re.match(r'(.+)_image_(\d+)\.(\w+)$', filename)
        if match:
            session_id = match.group(1)
            image_number = int(match.group(2))
            extension = match.group(3)
            
            # Calculate page numbers: image N -> pages (N*2+1) and (N*2+2)
            left_page = (image_number * 2) + 1
            right_page = (image_number * 2) + 2
            
            # Build split image keys
            left_key = f"{directory}/splitted/{session_id}_page_{left_page}.{extension}"
            right_key = f"{directory}/splitted/{session_id}_page_{right_page}.{extension}"
            
            return left_key, right_key
        
        # Try pattern 2: image_{number}.png (without session_id)
        match = re.match(r'image_(\d+)\.(\w+)$', filename)
        if match:
            image_number = int(match.group(1))
            extension = match.group(2)
            
            # Calculate page numbers: image 1 -> pages 1-2, image 2 -> pages 3-4, etc.
            # Formula: image N -> pages ((N-1)*2+1) and ((N-1)*2+2)
            left_page = ((image_number - 1) * 2) + 1
            right_page = ((image_number - 1) * 2) + 2
            
            # Build split image keys (replace 'full' with 'splitted' in directory)
            split_directory = directory.replace('/full', '/splitted')
            left_key = f"{split_directory}/page_{left_page}.{extension}"
            right_key = f"{split_directory}/page_{right_page}.{extension}"
            
            return left_key, right_key
        
        # If no pattern matches, raise error
        raise ValueError(f"Invalid filename pattern: {filename}. Expected format: '{{session_id}}_image_{{number}}.ext' or 'image_{{number}}.ext'")
    
    def _download_image_from_s3(self, s3_object_key: str) -> bytes:
        """Download image from S3 bucket"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_object_key)
            return response['Body'].read()
        except Exception as e:
            print(f"Error downloading from S3: {str(e)}")
            raise
    
    def _delete_old_image(self, s3_object_key: str) -> dict:
        """Delete the old image from S3"""
        try:
            result = delete_file_from_s3(bucket_name=self.bucket_name, object_name=s3_object_key)
            if result['success']:
                print(f"Deleted old image: {s3_object_key}")
            else:
                print(f"Failed to delete {s3_object_key}: {result['message']}")
            return result
        except Exception as e:
            print(f"Error deleting old image {s3_object_key}: {str(e)}")
            # Don't raise, just log - regeneration is more important
            return {'success': False, 'message': str(e)}
    
    def _generate_cover_image(
        self,
        prompt: str,
        story: str,
        reference_image_bytes: bytes,
        gender: str,
        age: int,
        image_style: str,
        full_s3_key: str
    ) -> PageImageUrls:
        """Generate a corrected cover image (no splitting) - 8.5" x 8.5" """
        try:
            # Create enhanced prompt with error correction instructions
            enhanced_prompt = self._create_correction_prompt(prompt, story, gender, age, image_style)
            
            # Encode reference image to base64
            reference_image_base64 = base64.b64encode(reference_image_bytes).decode('utf-8')
            
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Generate cover image: 8.5" x 8.5" at 300 DPI = 2550x2550 pixels
            width, height = 2550, 2550
            
            # Prepare payload with reference image
            payload = {
                'model': self.model,
                'prompt': enhanced_prompt,
                'width': width,
                'height': height,
                'watermark': False,
                'image_reference': [
                    {
                        'image': f'data:image/png;base64,{reference_image_base64}',
                        'weight': 0.8  # High weight to maintain similarity
                    }
                ]
            }
            
            # Call SeeDream API
            print(f"Calling SeeDream API to regenerate cover image...")
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract image URL from response
            image_url = result.get('data', [{}])[0].get('url') or result.get('image_url') or result.get('url')
            
            if not image_url:
                raise ValueError("No image URL in SeeDream API response")
            
            # Download the regenerated image
            print(f"Downloading regenerated cover image from: {image_url}")
            img_response = requests.get(image_url, timeout=60)
            img_response.raise_for_status()
            
            # Open and resize to exact dimensions if needed
            img = Image.open(io.BytesIO(img_response.content))
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Save cover image to BytesIO
            cover_img_buffer = io.BytesIO()
            img.save(cover_img_buffer, format='PNG', dpi=(300, 300))
            cover_img_buffer.seek(0)
            
            # Upload cover image to S3
            print(f"Uploading cover image to S3: {full_s3_key}")
            cover_upload_result = upload_file_object_to_s3(
                file_object=cover_img_buffer,
                bucket_name=self.bucket_name,
                object_name=full_s3_key
            )
            
            if not cover_upload_result['success']:
                raise ValueError(f"Failed to upload cover image to S3: {cover_upload_result['message']}")
            
            # Return PageImageUrls object with only fullPageUrl (no splitting for cover)
            return PageImageUrls(
                fullPageUrl=cover_upload_result['url'],
                leftUrl=None,
                rightUrl=None
            )
            
        except Exception as e:
            print(f"Error generating cover image: {str(e)}")
            raise
    
    def _generate_and_split_image(
        self,
        prompt: str,
        story: str,
        reference_image_bytes: bytes,
        gender: str,
        age: int,
        image_style: str,
        full_s3_key: str,
        left_s3_key: str,
        right_s3_key: str,
        page_number: int,
        font_size: int = 100,
        text_color: str = "white",
        dpi: int = 300
    ) -> PageImageUrls:
        """Generate a corrected full image, add text, split it, and upload all three versions"""
        try:
            # Create enhanced prompt with error correction instructions
            enhanced_prompt = self._create_correction_prompt(prompt, story, gender, age, image_style)
            
            # Encode reference image to base64
            reference_image_base64 = base64.b64encode(reference_image_bytes).decode('utf-8')
            
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Generate full double-page image: 17" x 8.5" at 300 DPI = 5100x2550 pixels
            width, height = 5100, 2550
            
            # Prepare payload with reference image
            payload = {
                'model': self.model,
                'prompt': enhanced_prompt,
                'width': width,
                'height': height,
                'watermark': False,
                'image_reference': [
                    {
                        'image': f'data:image/png;base64,{reference_image_base64}',
                        'weight': 0.8  # High weight to maintain similarity
                    }
                ]
            }
            
            # Call SeeDream API
            print(f"Calling SeeDream API to regenerate image...")
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract image URL from response
            image_url = result.get('data', [{}])[0].get('url') or result.get('image_url') or result.get('url')
            
            if not image_url:
                raise ValueError("No image URL in SeeDream API response")
            
            # Download the regenerated image
            print(f"Downloading regenerated image from: {image_url}")
            img_response = requests.get(image_url, timeout=60)
            img_response.raise_for_status()
            
            # Open and resize to exact dimensions if needed
            img = Image.open(io.BytesIO(img_response.content))
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Add text to the image if story is provided
            if story and story.strip():
                print(f"Adding text to regenerated image...")
                try:
                    # Use image analysis to determine best text placement
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    img_bytes = img_buffer.read()
                    
                    # Get text placement recommendation from image analysis (async function)
                    import asyncio
                    text_recommendation = asyncio.run(get_text_placement_recommendation(
                        img_bytes,
                        story,
                        font_size
                    ))
                    
                    # Open image for drawing
                    draw = ImageDraw.Draw(img)
                    
                    # Load font
                    from pathlib import Path
                    font_path = Path(__file__).resolve().parents[3] / "fonts" / "Comic_Relief" / "ComicRelief-Regular.ttf"
                    try:
                        if font_path.exists():
                            font = ImageFont.truetype(str(font_path), font_size)
                            print(f"Loaded Comic Relief font at size {font_size}")
                        else:
                            font = ImageFont.load_default()
                            print(f"Font not found, using default")
                    except Exception as font_err:
                        font = ImageFont.load_default()
                        print(f"Font load error: {font_err}, using default")
                    
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
                    
                    print(f"Text successfully added to regenerated image")
                        
                except Exception as text_error:
                    print(f"Error adding text to regenerated image: {str(text_error)}")
                    import traceback
                    traceback.print_exc()
                    # Continue without text if it fails
            
            # Save full image to BytesIO
            full_img_buffer = io.BytesIO()
            img.save(full_img_buffer, format='PNG', dpi=(300, 300))
            full_img_buffer.seek(0)
            
            # Upload full image to S3
            print(f"Uploading full image to S3: {full_s3_key}")
            full_upload_result = upload_file_object_to_s3(
                file_object=full_img_buffer,
                bucket_name=self.bucket_name,
                object_name=full_s3_key
            )
            
            if not full_upload_result['success']:
                raise ValueError(f"Failed to upload full image to S3: {full_upload_result['message']}")
            
            # Split the image in the middle
            # Left half: 0 to 2550 pixels (8.5" x 8.5")
            # Right half: 2550 to 5100 pixels (8.5" x 8.5")
            middle_x = width // 2  # 2550
            
            left_half = img.crop((0, 0, middle_x, height))
            right_half = img.crop((middle_x, 0, width, height))
            
            # Calculate page numbers based on page_number parameter
            # page_number is the image number (e.g., 1, 2, 3...)
            # Left page = (page_number * 2) + 1
            # Right page = (page_number * 2) + 2
            left_page_num = (page_number * 2) + 1
            right_page_num = (page_number * 2) + 2
            
            print(f"Splitting image {page_number} into pages {left_page_num} and {right_page_num}")
            
            # Save left half
            left_img_buffer = io.BytesIO()
            left_half.save(left_img_buffer, format='PNG', dpi=(300, 300))
            left_img_buffer.seek(0)
            
            # Upload left half to S3
            print(f"Uploading left image to S3: {left_s3_key}")
            left_upload_result = upload_file_object_to_s3(
                file_object=left_img_buffer,
                bucket_name=self.bucket_name,
                object_name=left_s3_key
            )
            
            if not left_upload_result['success']:
                raise ValueError(f"Failed to upload left image to S3: {left_upload_result['message']}")
            
            # Save right half
            right_img_buffer = io.BytesIO()
            right_half.save(right_img_buffer, format='PNG', dpi=(300, 300))
            right_img_buffer.seek(0)
            
            # Upload right half to S3
            print(f"Uploading right image to S3: {right_s3_key}")
            right_upload_result = upload_file_object_to_s3(
                file_object=right_img_buffer,
                bucket_name=self.bucket_name,
                object_name=right_s3_key
            )
            
            if not right_upload_result['success']:
                raise ValueError(f"Failed to upload right image to S3: {right_upload_result['message']}")
            
            # Return PageImageUrls object
            return PageImageUrls(
                fullPageUrl=full_upload_result['url'],
                leftUrl=left_upload_result['url'],
                rightUrl=right_upload_result['url']
            )
            
        except Exception as e:
            print(f"Error generating and splitting image: {str(e)}")
            raise
    
    def _update_pdf_with_new_pages(
        self,
        pdf_url: str,
        page_image_urls: PageImageUrls,
        page_number: int,
        is_cover: bool
    ) -> str:
        """Download PDF, replace regenerated page(s), and re-upload"""
        try:
            # Extract S3 key from PDF URL
            pdf_s3_key = self._extract_s3_key_from_url(pdf_url)
            print(f"Downloading PDF from S3: {pdf_s3_key}")
            
            # Download existing PDF from S3
            pdf_bytes = self._download_image_from_s3(pdf_s3_key)
            pdf_buffer = io.BytesIO(pdf_bytes)
            
            # Read the existing PDF
            pdf_reader = PdfReader(pdf_buffer)
            pdf_writer = PdfWriter()
            
            # Download the new regenerated image(s)
            print(f"Downloading regenerated images for PDF update...")
            
            if is_cover:
                # Cover page (page 0 in PDF)
                new_img_response = requests.get(page_image_urls.fullPageUrl, timeout=60)
                new_img_response.raise_for_status()
                new_img = Image.open(io.BytesIO(new_img_response.content))
                
                # Convert to RGB if needed
                if new_img.mode != 'RGB':
                    new_img = new_img.convert('RGB')
                
                # Resize to 8.5" x 8.5" at 72 DPI for PDF
                new_img = new_img.resize((612, 612), Image.Resampling.LANCZOS)
                
                # Create new PDF page with the image
                new_page_buffer = io.BytesIO()
                new_page_canvas = pdf_canvas.Canvas(new_page_buffer, pagesize=(612, 612))
                
                img_buffer = io.BytesIO()
                new_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                img_reader = ImageReader(img_buffer)
                
                new_page_canvas.drawImage(img_reader, 0, 0, width=612, height=612)
                new_page_canvas.save()
                new_page_buffer.seek(0)
                
                # Read the new page
                new_page_pdf = PdfReader(new_page_buffer)
                
                # Copy all pages, replacing page 0
                for i in range(len(pdf_reader.pages)):
                    if i == 0:
                        pdf_writer.add_page(new_page_pdf.pages[0])
                    else:
                        pdf_writer.add_page(pdf_reader.pages[i])
                
                print(f"Replaced cover page (page 0) in PDF")
            
            else:
                # Regular page - download both left and right split images
                left_img_response = requests.get(page_image_urls.leftUrl, timeout=60)
                left_img_response.raise_for_status()
                left_img = Image.open(io.BytesIO(left_img_response.content))
                
                right_img_response = requests.get(page_image_urls.rightUrl, timeout=60)
                right_img_response.raise_for_status()
                right_img = Image.open(io.BytesIO(right_img_response.content))
                
                # Convert to RGB if needed
                if left_img.mode != 'RGB':
                    left_img = left_img.convert('RGB')
                if right_img.mode != 'RGB':
                    right_img = right_img.convert('RGB')
                
                # Resize to 8.5" x 8.5" at 72 DPI for PDF
                left_img = left_img.resize((612, 612), Image.Resampling.LANCZOS)
                right_img = right_img.resize((612, 612), Image.Resampling.LANCZOS)
                
                # Calculate PDF page indices
                # page_number is the image number (e.g., 1, 2, 3...)
                # PDF structure: index 0 = cover, indices 1-2 = logo pages, indices 3+ = story pages
                # For page_number=1: should replace PDF indices 3 and 4
                # For page_number=2: should replace PDF indices 5 and 6
                # Formula: left_pdf_index = page_number * 2 + 1, right_pdf_index = page_number * 2 + 2
                left_pdf_index = page_number * 2 + 1
                right_pdf_index = page_number * 2 + 2
                
                print(f"Replacing PDF indices: {left_pdf_index} and {right_pdf_index}")
                
                # Create new PDF pages with the images
                left_page_buffer = io.BytesIO()
                left_page_canvas = pdf_canvas.Canvas(left_page_buffer, pagesize=(612, 612))
                left_img_buffer = io.BytesIO()
                left_img.save(left_img_buffer, format='PNG')
                left_img_buffer.seek(0)
                left_img_reader = ImageReader(left_img_buffer)
                left_page_canvas.drawImage(left_img_reader, 0, 0, width=612, height=612)
                left_page_canvas.save()
                left_page_buffer.seek(0)
                
                right_page_buffer = io.BytesIO()
                right_page_canvas = pdf_canvas.Canvas(right_page_buffer, pagesize=(612, 612))
                right_img_buffer = io.BytesIO()
                right_img.save(right_img_buffer, format='PNG')
                right_img_buffer.seek(0)
                right_img_reader = ImageReader(right_img_buffer)
                right_page_canvas.drawImage(right_img_reader, 0, 0, width=612, height=612)
                right_page_canvas.save()
                right_page_buffer.seek(0)
                
                # Read the new pages
                left_new_page = PdfReader(left_page_buffer)
                right_new_page = PdfReader(right_page_buffer)
                
                # Copy all pages, replacing the two pages
                for i in range(len(pdf_reader.pages)):
                    if i == left_pdf_index:
                        pdf_writer.add_page(left_new_page.pages[0])
                    elif i == right_pdf_index:
                        pdf_writer.add_page(right_new_page.pages[0])
                    else:
                        pdf_writer.add_page(pdf_reader.pages[i])
                
                print(f"Replaced PDF indices {left_pdf_index} and {right_pdf_index} successfully")
            
            # Write updated PDF to buffer
            updated_pdf_buffer = io.BytesIO()
            pdf_writer.write(updated_pdf_buffer)
            updated_pdf_buffer.seek(0)
            
            # Delete old PDF from S3
            print(f"Deleting old PDF from S3: {pdf_s3_key}")
            delete_result = delete_file_from_s3(bucket_name=self.bucket_name, object_name=pdf_s3_key)
            if delete_result['success']:
                print(f"Old PDF deleted successfully")
            else:
                print(f"Warning: Failed to delete old PDF: {delete_result['message']}")
            
            # Upload updated PDF to S3
            print(f"Uploading updated PDF to S3: {pdf_s3_key}")
            upload_result = upload_file_object_to_s3(
                file_object=updated_pdf_buffer,
                bucket_name=self.bucket_name,
                object_name=pdf_s3_key
            )
            
            if not upload_result['success']:
                raise ValueError(f"Failed to upload updated PDF to S3: {upload_result['message']}")
            
            print(f"Updated PDF uploaded successfully: {upload_result['url']}")
            return upload_result['url']
            
        except Exception as e:
            print(f"Error updating PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_correction_prompt(self, prompt: str, story: str, gender: str, age: int, image_style: str) -> str:
        """Create an enhanced prompt with error correction instructions"""
        correction_instructions = (
            "CRITICAL: Regenerate this image fixing any errors or inconsistencies while keeping everything else identical. "
            "Fix issues like: incorrect gender representation, anatomical errors (especially hands, fingers, limbs), "
            "inconsistent character features, unwanted objects (books, text, labels), unnatural proportions, "
            "or any other visual mistakes. Maintain the exact same composition, scene, colors, style, and character design. "
            "Only correct the errors - do not change what is correct. "
        )
        
        gender_description = "boy" if gender.lower() == "male" else "girl"
        age_description = f"{age}-year-old {gender_description}"
        
        enhanced_prompt = (
            f"{correction_instructions}"
            f"\n\nOriginal Prompt: {prompt}"
            f"\n\nCharacter: {age_description}, illustrated in {image_style} style. "
            f"Ensure the character's gender is clearly {gender_description}, with appropriate features and appearance. "
            f"Verify all anatomical details are correct and natural-looking."
        )
        
        return enhanced_prompt
    


