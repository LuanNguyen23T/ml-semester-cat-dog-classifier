import os
import uuid
from pathlib import Path
import cloudinary
import cloudinary.uploader
from app.core.config import settings

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "static" / "uploads"

if settings.CLOUDINARY_CLOUD_NAME and settings.CLOUDINARY_API_KEY:
    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_API_SECRET
    )

def upload_image_to_cloudinary(file_content: bytes, filename: str, user_id: str = "anonymous") -> str:
    """Uploads an image to Cloudinary and returns the secure URL.
    Falls back to local file storage if Cloudinary credentials are not configured.
    """
    if settings.CLOUDINARY_CLOUD_NAME and settings.CLOUDINARY_API_KEY:
        folder_path = f"pet-breed-classifier/users/{user_id}"
        response = cloudinary.uploader.upload(
            file_content,
            folder=folder_path,
            resource_type="image"
        )
        return response.get("secure_url")

    # Local storage fallback
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    target_path = UPLOAD_DIR / unique_name
    with open(target_path, "wb") as f:
        f.write(file_content)
    return f"http://localhost:8000/static/uploads/{unique_name}"

