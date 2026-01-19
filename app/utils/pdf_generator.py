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
    """Generate a PDF book with cover, logo, and all pages
    Each page is 8.5\" x 8.5\" at 300 DPI"""
    try:
        # Page size: 8.5" x 8.5" at 72 points per inch
        page_size = (8.5 * 72, 8.5 * 72)  # 612 x 612 points
        
        # Create PDF in memory
        pdf_buffer = io.BytesIO()
        c = pdf_canvas.Canvas(pdf_buffer, pagesize=page_size)
        
        # Sort pages by number
        sorted_pages = sorted(
            [(k, v) for k, v in image_bytes.items() if k.startswith('page ')],
            key=lambda x: int(x[0].split()[1])
        )
        
        print(f"\nGenerating PDF with {len(sorted_pages)} pages + logo page...")
        
        # Add each page to PDF
        for page_key, page_bytes in sorted_pages:
            try:
                # Open image from bytes
                img = Image.open(io.BytesIO(page_bytes))
                
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
                
                print(f"  - Added {page_key} to PDF")
                
                # Add logo page after cover (page 0)
                if page_key == 'page 0':
                    # Create a simple logo page (you can customize this)
                    c.setFont("Helvetica-Bold", 24)
                    c.drawCentredString(306, 306, "FaceToon")
                    c.showPage()
                    print(f"  - Added logo page after cover")
                
            except Exception as e:
                print(f"Error adding {page_key} to PDF: {str(e)}")
                continue
        
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
