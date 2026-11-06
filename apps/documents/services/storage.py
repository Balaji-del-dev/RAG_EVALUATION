"""Original files live in MinIO. PostgreSQL stores only the object key."""

import io
from functools import lru_cache

from minio import Minio
from minio.error import S3Error
from urllib3.exceptions import HTTPError

from apps.common.exceptions import ServiceUnavailable
from config.env import env


@lru_cache(maxsize=1)
def _client() -> Minio:
    return Minio(env.MINIO_ENDPOINT, access_key=env.MINIO_ACCESS_KEY, secret_key=env.MINIO_SECRET_KEY, secure=False)


def _ensure_bucket(client: Minio) -> None:
    if not client.bucket_exists(env.MINIO_BUCKET):
        client.make_bucket(env.MINIO_BUCKET)


def put_object(key: str, data: bytes, content_type: str) -> None:
    try:
        client = _client()
        _ensure_bucket(client)
        client.put_object(env.MINIO_BUCKET, key, io.BytesIO(data), length=len(data), content_type=content_type)
    except (S3Error, HTTPError, OSError) as exc:
        raise ServiceUnavailable(f"MinIO write failed: {exc}") from exc


def get_object(key: str) -> bytes:
    try:
        response = _client().get_object(env.MINIO_BUCKET, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
    except (S3Error, HTTPError, OSError) as exc:
        raise ServiceUnavailable(f"MinIO read failed: {exc}") from exc


def delete_object(key: str) -> None:
    try:
        _client().remove_object(env.MINIO_BUCKET, key)
    except (S3Error, HTTPError, OSError) as exc:
        raise ServiceUnavailable(f"MinIO delete failed: {exc}") from exc
