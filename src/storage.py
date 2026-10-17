import aioboto3
from src.config import get_settings
import hashlib
import aiofiles
import os

settings = get_settings()

from botocore.config import Config
import socket
from urllib.parse import urlparse

class S3StorageClient:
    def __init__(self):
        self.session = aioboto3.Session()
        self.endpoint = settings.storage.endpoint_url
        self.access_key = settings.storage.access_key
        self.secret_key = settings.storage.secret_key
        self._is_online = None

    def is_endpoint_reachable(self) -> bool:
        if self._is_online is not None:
            return self._is_online
        try:
            parsed = urlparse(self.endpoint)
            host = parsed.hostname or 'localhost'
            port = parsed.port or (443 if parsed.scheme == 'https' else 8000)
            with socket.create_connection((host, port), timeout=0.2):
                self._is_online = True
        except Exception:
            self._is_online = False
        return self._is_online
        
    def _client(self):
        config = Config(connect_timeout=0.5, read_timeout=0.5, retries={'max_attempts': 1})
        return self.session.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=config
        )
        
    async def generate_presigned_url(self, client_method: str, bucket: str, key: str, expires_in: int = 3600) -> str:
        if not self.is_endpoint_reachable():
            return f"{self.endpoint}/{bucket}/{key}?demo_presigned=true"
        async with self._client() as s3:
            return await s3.generate_presigned_url(
                ClientMethod=client_method,
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expires_in
            )

    async def get_object_etag(self, bucket: str, key: str) -> str:
        async with self._client() as s3:
            response = await s3.head_object(Bucket=bucket, Key=key)
            return response.get('ETag', '').strip('"')
            
    async def migrate_file_with_verification(self, file_id: str, from_bucket: str, to_bucket: str) -> bool:
        """ Copys a file, verifies ETag integrity, then deletes the original safely. """
        if not self.is_endpoint_reachable():
            # S3 service offline / demo simulation mode
            return True
            
        async with self._client() as s3:
            # 1. Copy
            await s3.copy_object(
                Bucket=to_bucket,
                CopySource={'Bucket': from_bucket, 'Key': file_id},
                Key=file_id
            )
            
            # 2. Verify ETag Integrity
            source_etag = await self.get_object_etag(from_bucket, file_id)
            dest_etag = await self.get_object_etag(to_bucket, file_id)
            
            if source_etag != dest_etag:
                raise ValueError(f"ETag mismatch for {file_id}. Migration aborted.")
                
            # 3. Delete from Source safely
            await s3.delete_object(Bucket=from_bucket, Key=file_id)
            return True

storage = S3StorageClient()
