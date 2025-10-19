import boto3
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "titweng-cattle-images")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def upload_image_to_s3(file_content: bytes, file_name: str, content_type: str = "image/jpeg") -> str:
    """Upload image to S3 and return public URL"""
    try:
        # Upload file to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"cow_images/{file_name}",
            Body=file_content,
            ContentType=content_type,
            ACL='public-read'  # Make publicly accessible
        )
        
        # Return public URL
        public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/cow_images/{file_name}"
        return public_url
        
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        return None

def upload_receipt_to_s3(file_content: bytes, file_name: str) -> str:
    """Upload PDF receipt to S3 and return public URL"""
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"receipts/{file_name}",
            Body=file_content,
            ContentType="application/pdf",
            ACL='public-read'
        )
        
        public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/receipts/{file_name}"
        return public_url
        
    except ClientError as e:
        print(f"Error uploading receipt to S3: {e}")
        return None