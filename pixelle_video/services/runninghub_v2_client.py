"""
RunningHub V2 AI App API client.

Supports the /openapi/v2/run/ai-app/{appId} endpoint for RunningHub AI Apps.
Unlike the v1 API, v2 does not require fetching workflow JSON first —
we directly POST parameters and poll for results.
"""

import asyncio
import os
import ssl
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import certifi
from loguru import logger


# Node class types that require file upload (image/mask inputs)
_UPLOAD_NODE_TYPES = {"LoadImage", "LoadImageMask", "VHS_LoadVideo", "LoadAudio"}


class RunningHubV2Client:
    """Lightweight async client for RunningHub v2 AI App API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 300,
        poll_interval: int = 3,
        max_wait_time: Optional[int] = None,
        instance_type: str = "default",
    ):
        self.api_key = api_key or os.getenv("RUNNINGHUB_API_KEY")
        self.base_url = (base_url or os.getenv("RUNNINGHUB_BASE_URL", "https://www.runninghub.ai")).rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time  # None = unlimited
        self.instance_type = instance_type
        self._session: Optional[aiohttp.ClientSession] = None

        if not self.api_key:
            raise ValueError("RunningHub API key is required")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout_config,
                connector=connector,
                trust_env=True,
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # File upload
    # ------------------------------------------------------------------

    async def upload_file(self, file_path: str) -> str:
        """Upload a file to RunningHub and return its fileName reference."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            file_content = f.read()

        filename = Path(file_path).name
        data = aiohttp.FormData()
        data.add_field("apiKey", self.api_key)
        data.add_field("file", file_content, filename=filename)

        session = await self._get_session()
        url = f"{self.base_url}/task/openapi/upload"

        async with session.post(url, data=data) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Upload HTTP {resp.status}: {text}")
            result = await resp.json()
            if result.get("code") != 0:
                raise Exception(f"Upload failed: {result.get('msg', 'Unknown error')}")
            file_name = result.get("data", {}).get("fileName", "")
            if not file_name:
                raise Exception("No fileName in upload response")
            logger.info(f"File uploaded: {file_path} -> {file_name}")
            return file_name

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    async def submit_task(
        self,
        app_id: str,
        node_info_list: List[Dict[str, Any]],
        instance_type: Optional[str] = None,
    ) -> str:
        """Submit a task to RunningHub v2 API and return taskId."""
        session = await self._get_session()
        url = f"{self.base_url}/openapi/v2/run/ai-app/{app_id}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "nodeInfoList": node_info_list,
            "instanceType": instance_type or self.instance_type,
            "usePersonalQueue": "false",
        }

        logger.info(f"Submitting v2 task to app={app_id} with {len(node_info_list)} node changes")

        async with session.post(url, headers=headers, json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Task submission HTTP {resp.status}: {text}")
            result = await resp.json()
            task_id = result.get("taskId", "")
            status = result.get("status", "")
            if not task_id:
                raise Exception(f"Task submission failed: {result}")
            logger.info(f"V2 task submitted: app={app_id}, taskId={task_id}, status={status}")
            return task_id

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def query_task(self, task_id: str) -> Dict[str, Any]:
        """Query task status from RunningHub v2 API."""
        session = await self._get_session()
        url = f"{self.base_url}/openapi/v2/query?taskId={task_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Query HTTP {resp.status}: {text}")
            return await resp.json()

    async def wait_for_completion(self, task_id: str) -> Dict[str, Any]:
        """Poll until task completes and return the final result."""
        start = time.time()
        while True:
            if self.max_wait_time and (time.time() - start) >= self.max_wait_time:
                raise TimeoutError(f"Task {task_id} timed out after {self.max_wait_time}s")

            result = await self.query_task(task_id)
            status = result.get("status", "")

            if status == "SUCCESS":
                logger.info(f"Task {task_id} completed successfully")
                return result
            elif status == "FAILED":
                error_msg = result.get("errorMessage") or result.get("msg") or "Unknown error"
                raise Exception(f"Task {task_id} failed: {error_msg}")
            elif status in ("QUEUED", "RUNNING"):
                logger.debug(f"Task {task_id} status: {status}, polling in {self.poll_interval}s...")
                await asyncio.sleep(self.poll_interval)
            else:
                logger.warning(f"Task {task_id} unknown status '{status}', retrying...")
                await asyncio.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # High-level execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        app_id: str,
        params: Dict[str, Any],
        param_mappings: List[Dict[str, Any]],
        instance_type: Optional[str] = None,
    ) -> List[str]:
        """Execute a v2 workflow and return list of result image URLs.

        Args:
            app_id: RunningHub AI App ID.
            params: Simple params dict, e.g.
                {"image": "/path/to/img.png", "prompt": "a cat", "mask": "/path/to/mask.png"}.
            param_mappings: Node mappings from the workflow JSON wrapper.
            instance_type: Override instance type.

        Returns:
            List of result image URLs.
        """
        node_info_list: List[Dict[str, Any]] = []

        for mapping in param_mappings:
            param_name = mapping["param_name"]
            if param_name not in params:
                continue

            param_value = params[param_name]
            node_id = mapping["node_id"]
            field_name = mapping["input_field"]
            node_class_type = mapping.get("node_class_type", "")

            # Upload local files that map to media-loading nodes
            if node_class_type in _UPLOAD_NODE_TYPES and isinstance(param_value, str):
                if os.path.exists(param_value):
                    param_value = await self.upload_file(param_value)

            node_info_list.append({
                "nodeId": node_id,
                "fieldName": field_name,
                "fieldValue": str(param_value),
            })

        if not node_info_list:
            raise ValueError("No parameters matched the param_mappings")

        logger.info(f"Executing v2 workflow: app={app_id}, nodes={len(node_info_list)}")
        task_id = await self.submit_task(app_id, node_info_list, instance_type)
        result = await self.wait_for_completion(task_id)

        # Extract image URLs from results
        image_urls: List[str] = []
        for item in (result.get("results") or []):
            if isinstance(item, dict):
                url = item.get("url")
                if url:
                    image_urls.append(url)

        logger.info(f"V2 task {task_id} returned {len(image_urls)} image(s)")
        return image_urls
