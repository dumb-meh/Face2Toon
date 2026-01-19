import io
import os
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
from typing import Dict, Optional
from app.utils.upload_to_bucket import upload_file_object_to_s3
import traceback


def generate_pdf(
    image_bytes: Dict[str, bytes],
    book_uuid: str,
    session_id: str,
    upload_to_s3: bool = False
) -> Optional[str]:
    """Generate a PDF book with cover, logo pages, and all pages
    Page order: Cover (page 0), Logo.png, Logo_2.png, then story pages (1-22, 12, 13, 14)
    Each page is 8.5\" x 8.5\" at 300 DPI"""
    try:
        # Page size: 8.5" x 8.5" at 72 points per inch
        page_size = (8.5 * 72, 8.5 * 72)  # 612 x 612 points
        
        # Create PDF in memory
        pdf_buffer = io.BytesIO()
        c = pdf_canvas.Canvas(pdf_buffer, pagesize=page_size)
        
        # Sort pages by number, handling 'page last page' specially
        def get_page_sort_key(item):
            page_key = item[0]
            if page_key == 'page last page':
                return 999  # Sort last
            elif page_key.startswith('page ') and page_key.split()[1].isdigit():
                return int(page_key.split()[1])
            else:
                return 0
        
        sorted_pages = sorted(
            [(k, v) for k, v in image_bytes.items() if k.startswith('page ')],
            key=get_page_sort_key
        )
        
        print(f"\nGenerating PDF with {len(sorted_pages)} pages + 2 logo pages...")
        
        # Helper function to add an image to PDF
        def add_image_to_pdf(img_source, page_name):
            """Add image to PDF from either bytes or file path"""
            try:
                if isinstance(img_source, bytes):
                    img = Image.open(io.BytesIO(img_source))
                else:
                    # It's a file path
                    img = Image.open(img_source)
                
                # Convert to RGB if needed (PDF doesn't support RGBA)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to 8.5\" x 8.5\" at 72 DPI for PDF
                img = img.resize((612, 612), Image.Resampling.LANCZOS)
                
                # Convert to ImageReader for reportlab
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                img_reader = ImageReader(img_buffer)
                
                # Draw image on PDF page
                c.drawImage(img_reader, 0, 0, width=612, height=612)
                c.showPage()
                
                print(f"  - Added {page_name} to PDF")
                return True
            except Exception as e:
                print(f"Error adding {page_name} to PDF: {str(e)}")
                return False
        
        # Page 1: Add cover (page 0)
        if 'page 0' in image_bytes:
            add_image_to_pdf(image_bytes['page 0'], 'page 0 (cover)')
        
        # Page 2: Add Logo.png (in Docker, logos are at /app/Logo.png)
        logo_path = '/app/Logo.png' if os.path.exists('/app/Logo.png') else 'Logo.png'
        if os.path.exists(logo_path):
            add_image_to_pdf(logo_path, 'Logo.png (page 2)')
        else:
            print(f"  - Warning: Logo.png not found at {logo_path}")
            print(f"  - Tried paths: /app/Logo.png, Logo.png")
        
        # Page 3: Add Logo_2.png (in Docker, logos are at /app/Logo_2.png)
        logo2_path = '/app/Logo_2.png' if os.path.exists('/app/Logo_2.png') else 'Logo_2.png'
        if os.path.exists(logo2_path):
            add_image_to_pdf(logo2_path, 'Logo_2.png (page 3)')
        else:
            print(f"  - Warning: Logo_2.png not found at {logo2_path}")
            print(f"  - Tried paths: /app/Logo_2.png, Logo_2.png")
        
        # Pages 4+: Add story pages (page 1 through page 22, then page 12, page 13, page 14)
        for page_key, page_bytes in sorted_pages:
            if page_key != 'page 0':  # Skip cover since we already added it
                add_image_to_pdf(page_bytes, page_key)
        
        # Save PDF
        c.save()
        pdf_buffer.seek(0)
        
        # Upload to S3 or save locally
        if upload_to_s3:
            pdf_object_name = f"facetoon/{book_uuid}/book_{session_id}.pdf"
            upload_result = upload_file_object_to_s3(pdf_buffer, object_name=pdf_object_name)
            
            if upload_result['success']:
                print(f"\nPDF uploaded to S3: {upload_result['url']}")
                return upload_result['url']
            else:
                print(f"Failed to upload PDF to S3: {upload_result['message']}")
                return None
        else:
            # Save locally
            os.makedirs('uploads/generated_pdfs', exist_ok=True)
            pdf_filename = f"uploads/generated_pdfs/{session_id}_book.pdf"
            
            with open(pdf_filename, 'wb') as f:
                f.write(pdf_buffer.read())
            
            base_url = os.getenv('domain') or os.getenv('BASE_URL')
            base_url = base_url.rstrip('/')
            pdf_url = f"{base_url}/{pdf_filename}"
            
            print(f"\nPDF saved locally: {pdf_url}")
            return pdf_url
            
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        traceback.print_exc()
        return None
