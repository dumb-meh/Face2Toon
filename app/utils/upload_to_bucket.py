import boto3
import os
from botocore.exceptions import ClientError
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
    region_name=os.getenv('S3_REGION', 'eu-north-1')
)

# Get bucket name and endpoint from environment
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'mycvconnect')
S3_ENDPOINT = os.getenv('S3_ENDPOINT')

def upload_file_to_s3(file_path: str, bucket_name: str = None, object_name: str = None) -> dict:
    """
    Upload a file to an S3 bucket with public read access
    
    Args:
        file_path (str): Path to the file to upload
        bucket_name (str, optional): Name of the S3 bucket. If not specified, uses S3_BUCKET_NAME from env
        object_name (str, optional): S3 object name. If not specified, file_path basename is used
    
    Returns:
        dict: Dictionary containing success status, file URL, and message
    """
    # Use default bucket if not specified
    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME
    
    # If S3 object_name was not specified, use file_path basename
    if object_name is None:
        object_name = os.path.basename(file_path)
    
    try:
        # Upload the file with public-read ACL
        s3_client.upload_file(
            file_path, 
            bucket_name, 
            object_name,
            ExtraArgs={
                'ACL': 'public-read',
                'ContentType': get_content_type(file_path)
            }
        )
        
        # Construct the public URL
        region = os.getenv('S3_REGION', 'eu-north-1')
        file_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_name}"
        
        logger.info(f"File uploaded successfully: {file_url}")
        
        return {
            'success': True,
            'url': file_url,
            'message': 'File uploaded successfully'
        }
    
    except ClientError as e:
        error_message = f"Failed to upload file: {str(e)}"
        logger.error(error_message)
        return {
            'success': False,
            'url': None,
            'message': error_message
        }
    except FileNotFoundError:
        error_message = f"The file {file_path} was not found"
        logger.error(error_message)
        return {
            'success': False,
            'url': None,
            'message': error_message
        }
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(error_message)
        return {
            'success': False,
            'url': None,
            'message': error_message
        }


def upload_file_object_to_s3(file_object, bucket_name: str = None, object_name: str = None) -> dict:
    """
    Upload a file-like object to an S3 bucket with public read access
    
    Args:
        file_object: File-like object to upload
        bucket_name (str, optional): Name of the S3 bucket. If not specified, uses S3_BUCKET_NAME from env
        object_name (str): S3 object name
    
    Returns:
        dict: Dictionary containing success status, file URL, and message
    """
    # Use default bucket if not specified
    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME
    try:
        # Upload the file object with public-read ACL
        s3_client.upload_fileobj(
            file_object,
            bucket_name,
            object_name,
            ExtraArgs={
                'ACL': 'public-read',
                'ContentType': get_content_type(object_name)
            }
        )
        
        # Construct the public URL
        region = os.getenv('S3_REGION', 'eu-north-1')
        file_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_name}"
        
        logger.info(f"File object uploaded successfully: {file_url}")
        
        return {
            'success': True,
            'url': file_url,
            'message': 'File uploaded successfully'
        }
    
    except ClientError as e:
        error_message = f"Failed to upload file object: {str(e)}"
        logger.error(error_message)
        return {
            'success': False,
            'url': None,
            'message': error_message
        }
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(error_message)
        return {
            'success': False,
            'url': None,
            'message': error_message
        }


def get_content_type(file_path: str) -> str:
    """
    Determine the content type based on file extension
    
    Args:
        file_path (str): Path or name of the file
    
    Returns:
        str: MIME type of the file
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    content_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml',
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.json': 'application/json',
        '.mp4': 'video/mp4',
        '.mp3': 'audio/mpeg',
    }
    
    return content_types.get(extension, 'application/octet-stream')


def delete_file_from_s3(bucket_name: str = None, object_name: str = None) -> dict:
    """
    Delete a file from an S3 bucket
    
    Args:
        bucket_name (str, optional): Name of the S3 bucket. If not specified, uses S3_BUCKET_NAME from env
        object_name (str): S3 object name to delete
    
    Returns:
        dict: Dictionary containing success status and message
    """
    # Use default bucket if not specified
    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=object_name)
        
        logger.info(f"File deleted successfully: {object_name}")
        
        return {
            'success': True,
            'message': 'File deleted successfully'
        }
    
    except ClientError as e:
        error_message = f"Failed to delete file: {str(e)}"
        logger.error(error_message)
        return {
            'success': False,
            'message': error_message
        }
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(error_message)
        return {
            'success': False,
            'message': error_message
        }
