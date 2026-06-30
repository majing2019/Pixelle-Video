# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Doubao TTS Utility - Volcano Engine (火山引擎) TTS API

Calls the Volcano Engine TTS HTTP v1 API.
Authentication: simple Bearer token (console-issued access_token).
"""

import asyncio
import base64
import os
import random
import uuid
from typing import Optional

import httpx
from loguru import logger

# Retry configuration
_RETRY_COUNT = 3
_RETRY_BASE_DELAY = 1.0
_MAX_RETRY_DELAY = 10.0

# Proxy environment variable names to suppress
_PROXY_ENV_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy",
)

# Volcano Engine TTS API endpoint
_TTS_URL = "https://openspeech.bytedance.com/api/v1/tts"
_CLUSTER = "volcano_tts"


class _ProxySuppressor:
    """Context manager that temporarily removes proxy environment variables.

    Follows the same pattern as Edge TTS to prevent proxies from
    interfering with API calls.
    """

    def __init__(self):
        self._saved = {}

    def __enter__(self):
        for var in _PROXY_ENV_VARS:
            val = os.environ.pop(var, None)
            if val is not None:
                self._saved[var] = val

    def __exit__(self, *args):
        os.environ.update(self._saved)
        self._saved.clear()


def _build_headers(access_key: str) -> dict:
    """
    Build HTTP headers for Volcano Engine TTS API.

    Uses simple Bearer token authentication (access_key is the
    console-issued access token).

    Args:
        access_key: Access token (console-issued)

    Returns:
        Dictionary of HTTP headers
    """
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer;{access_key}",
    }


async def doubao_tts(
    text: str,
    app_id: str,
    access_key: str,
    secret_key: str = "",
    voice: str = "BV001_streaming",
    speed: float = 1.0,
    volume: float = 1.0,
    pitch_ratio: float = 1.0,
    audio_format: str = "mp3",
    output_path: Optional[str] = None,
    retry_count: int = _RETRY_COUNT,
) -> bytes:
    """
    Convert text to speech using Volcano Engine (Doubao) TTS API.

    Args:
        text: Text to synthesize
        app_id: Volcano Engine application ID
        access_key: Access token (console-issued)
        secret_key: Secret key (reserved for future API versions)
        voice: Voice type ID (e.g., BV001_streaming)
        speed: Speech speed (0.2-3.0, default 1.0)
        volume: Volume (0.5-2.0, default 1.0)
        pitch_ratio: Pitch ratio (reserved, not used in v1 API)
        audio_format: Output format (mp3, wav, ogg, pcm)
        output_path: Optional file path to save audio
        retry_count: Number of retries on failure

    Returns:
        Audio data as bytes

    Raises:
        ValueError: If credentials are missing
        RuntimeError: If API call fails after retries
    """
    if not app_id or not access_key:
        raise ValueError(
            "Doubao TTS requires app_id and access_key. "
            "Please configure comfyui.tts.doubao in config.yaml."
        )

    logger.debug(
        f"Calling Doubao TTS: voice={voice}, speed={speed}, "
        f"volume={volume}, format={audio_format}"
    )

    headers = _build_headers(access_key)
    request_id = f"req_{uuid.uuid4().hex[:16]}"

    payload = {
        "app": {
            "appid": app_id,
            "cluster": _CLUSTER,
            "token": access_key,
        },
        "user": {"uid": "pixelle_video"},
        "audio": {
            "voice_type": voice,
            "encoding": audio_format,
            "speed_ratio": speed,
            "loudness_ratio": volume,
            "rate": 24000,
        },
        "request": {
            "reqid": request_id,
            "text": text,
            "text_type": "plain",
            "operation": "query",
        },
    }

    last_error = None

    for attempt in range(retry_count + 1):
        if attempt > 0:
            exponential_delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            jitter = random.uniform(0, _RETRY_BASE_DELAY)
            retry_delay = min(exponential_delay + jitter, _MAX_RETRY_DELAY)
            logger.info(
                f"🔄 Retrying Doubao TTS (attempt {attempt + 1}/{retry_count + 1}) "
                f"after {retry_delay:.2f}s delay..."
            )
            await asyncio.sleep(retry_delay)

        try:
            timeout = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0)
            with _ProxySuppressor():
                async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                    response = await client.post(_TTS_URL, headers=headers, json=payload)

            if response.status_code != 200:
                last_error = RuntimeError(
                    f"Doubao TTS HTTP {response.status_code}: {response.text}"
                )
                if attempt >= retry_count:
                    raise last_error
                continue

            content_type = response.headers.get("Content-Type", "")

            # Binary audio response (with_frontend=0 or default)
            if "audio" in content_type or "octet-stream" in content_type:
                audio_data = response.content
                logger.info(
                    f"Generated {len(audio_data)} bytes of audio data (Doubao TTS)"
                )
                if output_path:
                    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(audio_data)
                    logger.info(f"Audio saved to: {output_path}")
                return audio_data

            # JSON response with base64-encoded audio
            try:
                result = response.json()
                code = result.get("code")
                if code == 3000 and result.get("data"):
                    audio_data = base64.b64decode(result["data"])
                    logger.info(
                        f"Generated {len(audio_data)} bytes of audio data (Doubao TTS, JSON)"
                    )
                    if output_path:
                        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                        with open(output_path, "wb") as f:
                            f.write(audio_data)
                        logger.info(f"Audio saved to: {output_path}")
                    return audio_data
                else:
                    last_error = RuntimeError(
                        f"Doubao TTS API error: code={code}, message={result.get('message')}"
                    )
                    if attempt >= retry_count:
                        raise last_error
                    continue
            except Exception:
                last_error = RuntimeError(
                    f"Doubao TTS returned unexpected response: {response.text[:200]}"
                )
                if attempt >= retry_count:
                    raise last_error
                continue

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            last_error = e
            if attempt >= retry_count:
                raise RuntimeError(
                    f"Doubao TTS network error after {retry_count + 1} attempts: {e}"
                )
            continue
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            logger.error(f"Doubao TTS error (non-retryable): {type(e).__name__} - {e}")
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Doubao TTS failed without error (unexpected)")
