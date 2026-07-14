"""
Image-to-Image Pipeline UI

Generates new images from reference images and optional prompts,
supporting RunningHub and API modes.
"""

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from loguru import logger

from web.i18n import tr
from web.pipelines.base import PipelineUI, register_pipeline_ui
from web.pipelines.api_workflows import (
    is_api_workflow,
    list_api_media_workflows,
    list_local_media_workflows,
    workflow_source_label,
    workflow_select_help,
)
from web.utils.async_helpers import run_async
from pixelle_video.utils.os_util import create_task_output_dir


# ---------------------------------------------------------------------------
# Size presets (same as text-to-image)
# ---------------------------------------------------------------------------

_SIZE_PRESETS = {
    "1:1 1024×1024": "1024*1024",
    "1:1 2048×2048": "2048*2048",
    "16:9 1920×1080": "1920*1080",
    "16:9 2560×1440": "2560*1440",
    "9:16 1080×1920": "1080*1920",
    "9:16 1440×2560": "1440*2560",
    "4:3 1920×1440": "1920*1440",
    "3:4 1440×1920": "1440*1920",
}

# Rhart image-to-image specific options
_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
_RESOLUTIONS = ["1k", "2k", "4k"]

# Workflows that support dual-image input
_DUAL_IMAGE_WORKFLOW_KEYWORDS = ("rhart", "img2img_g2")


class Img2ImgPipelineUI(PipelineUI):
    """UI for the Image-to-Image pipeline."""

    name = "img2img"
    icon = "🖌️"

    @property
    def display_name(self):  # type: ignore[override]
        return tr("pipeline.img2img.name")

    @property
    def description(self):  # type: ignore[override]
        return tr("pipeline.img2img.description")

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def render(self, pixelle_video: Any):
        left_col, mid_col, right_col = st.columns([1, 1, 1])

        with left_col:
            input_params = self._render_input()

        with mid_col:
            config_params = self._render_style_config(pixelle_video)

        with right_col:
            self._render_output(pixelle_video, {**input_params, **config_params})

    # ------------------------------------------------------------------
    # Left column – reference image + prompt input
    # ------------------------------------------------------------------

    def _render_input(self) -> dict:
        params: Dict[str, Any] = {}

        with st.container(border=True):
            st.markdown(f"**{tr('img2img.section.input')}**")

            # --- Image source selection ---
            source_key = "img2img_image_source"
            source_options = ["upload", "url"]
            if st.session_state.get(source_key) not in source_options:
                st.session_state.pop(source_key, None)

            image_source = st.radio(
                tr("img2img.image_source"),
                source_options,
                format_func=lambda s: (
                    tr("img2img.source_upload") if s == "upload"
                    else tr("img2img.source_url")
                ),
                horizontal=True,
                key=source_key,
            )

            reference_image_path: Optional[str] = None

            if image_source == "upload":
                uploaded_file = st.file_uploader(
                    tr("img2img.upload"),
                    type=["jpg", "jpeg", "png", "webp"],
                    accept_multiple_files=False,
                    help=tr("img2img.upload_help"),
                    key="img2img_upload",
                )
                if uploaded_file:
                    # Save uploaded file to temp directory
                    session_id = str(uuid.uuid4()).replace("-", "")[:12]
                    temp_dir = Path(f"temp/img2img_{session_id}")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    file_path = temp_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    reference_image_path = str(file_path.absolute())
            else:
                url_input = st.text_input(
                    tr("img2img.url_input"),
                    placeholder=tr("img2img.url_placeholder"),
                    key="img2img_url",
                )
                if url_input.strip():
                    reference_image_path = url_input.strip()

            # Preview uploaded/URL image
            if reference_image_path:
                with st.expander(tr("img2img.preview"), expanded=True):
                    if os.path.exists(reference_image_path):
                        st.image(reference_image_path, use_container_width=True)
                    elif reference_image_path.startswith("http"):
                        st.image(reference_image_path, use_container_width=True)

            params["reference_image_path"] = reference_image_path

            # --- Optional second image (image2: style/layout reference) ---
            with st.expander(tr("img2img.image2_section"), expanded=False):
                st.caption(tr("img2img.image2_help"))

                img2_source = st.radio(
                    tr("img2img.image2_source"),
                    ["upload", "url"],
                    format_func=lambda s: (
                        tr("img2img.source_upload") if s == "upload"
                        else tr("img2img.source_url")
                    ),
                    horizontal=True,
                    key="img2img_image2_source",
                )

                reference_image2_path: Optional[str] = None

                if img2_source == "upload":
                    uploaded_file2 = st.file_uploader(
                        tr("img2img.upload_image2"),
                        type=["jpg", "jpeg", "png", "webp"],
                        accept_multiple_files=False,
                        key="img2img_upload2",
                    )
                    if uploaded_file2:
                        session_id = str(uuid.uuid4()).replace("-", "")[:12]
                        temp_dir = Path(f"temp/img2img_ref2_{session_id}")
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        file_path = temp_dir / uploaded_file2.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file2.getbuffer())
                        reference_image2_path = str(file_path.absolute())
                else:
                    url_input2 = st.text_input(
                        tr("img2img.url_input_image2"),
                        placeholder=tr("img2img.url_placeholder"),
                        key="img2img_url2",
                    )
                    if url_input2.strip():
                        reference_image2_path = url_input2.strip()

                if reference_image2_path:
                    if os.path.exists(reference_image2_path):
                        st.image(reference_image2_path, use_container_width=True)
                    elif reference_image2_path.startswith("http"):
                        st.image(reference_image2_path, use_container_width=True)

                params["reference_image2_path"] = reference_image2_path

            # --- Prompt (optional) ---
            prompt = st.text_area(
                tr("img2img.prompt"),
                placeholder=tr("img2img.prompt_placeholder"),
                height=150,
                help=tr("img2img.prompt_help"),
                key="img2img_prompt",
            )

            with st.expander(tr("img2img.negative_prompt"), expanded=False):
                negative_prompt = st.text_area(
                    tr("img2img.negative_prompt_label"),
                    placeholder="low quality, blurry, distorted, …",
                    height=80,
                    key="img2img_negative_prompt",
                )

            # --- Strength slider ---
            strength = st.slider(
                tr("img2img.strength"),
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help=tr("img2img.strength_help"),
                key="img2img_strength",
            )

            n_images = st.number_input(
                tr("img2img.count"),
                min_value=1,
                max_value=4,
                value=1,
                step=1,
                key="img2img_count",
            )

            params["prompt"] = prompt
            params["negative_prompt"] = negative_prompt.strip() or None
            params["strength"] = strength
            params["n_images"] = int(n_images)
            params["image_source"] = image_source

        return params

    # ------------------------------------------------------------------
    # Middle column – style config (RunningHub / API)
    # ------------------------------------------------------------------

    def _render_style_config(self, pixelle_video: Any) -> dict:
        params: Dict[str, Any] = {}

        with st.container(border=True):
            st.markdown(f"**{tr('img2img.section.config')}**")

            # --- mode selection (runninghub / api) ---
            source_options = ["runninghub", "api"]
            source_key = "img2img_workflow_source"
            if st.session_state.get(source_key) not in source_options:
                st.session_state.pop(source_key, None)

            workflow_source = st.radio(
                tr("img2img.generation_source"),
                source_options,
                format_func=workflow_source_label,
                horizontal=True,
                key=source_key,
            )

            # --- workflows / models ---
            if workflow_source == "runninghub":
                workflows = list_local_media_workflows(
                    pixelle_video,
                    "image",
                    "runninghub",
                    key_contains="img2img",
                )
            else:
                workflows = list_api_media_workflows(pixelle_video, "image")

            workflow_options = [wf["display_name"] for wf in workflows]
            workflow_keys = [wf["key"] for wf in workflows]

            workflow_display = st.selectbox(
                tr("img2img.workflow_select"),
                workflow_options if workflow_options else [tr("img2img.no_workflow")],
                index=0,
                key="img2img_workflow_select",
                help=workflow_select_help(),
            )

            workflow_key: Optional[str] = None
            workflow_info: Optional[dict] = None
            if workflow_options:
                idx = workflow_options.index(workflow_display)
                workflow_key = workflow_keys[idx]
                workflow_info = workflows[idx]
            else:
                st.warning(tr("img2img.no_workflow_warning"))

            params["workflow_key"] = workflow_key
            params["workflow_source"] = workflow_source
            params["workflow_info"] = workflow_info

            # --- detect dual-image workflow ---
            is_dual_image = (
                workflow_key
                and any(kw in workflow_key.lower() for kw in _DUAL_IMAGE_WORKFLOW_KEYWORDS)
            )

            # --- size presets (single-image workflows) / aspectRatio+resolution (dual-image) ---
            if is_dual_image:
                aspect_ratio = st.selectbox(
                    tr("img2img.aspect_ratio"),
                    _ASPECT_RATIOS,
                    index=2,  # default 9:16
                    key="img2img_aspect_ratio",
                    help=tr("img2img.aspect_ratio_help"),
                )
                params["aspect_ratio"] = aspect_ratio

                resolution = st.selectbox(
                    tr("img2img.resolution"),
                    _RESOLUTIONS,
                    index=0,  # default 1k
                    key="img2img_resolution",
                    help=tr("img2img.resolution_help"),
                )
                params["resolution"] = resolution
                params["size"] = None  # not used for dual-image workflows
            else:
                size_label = st.selectbox(
                    tr("img2img.size"),
                    list(_SIZE_PRESETS.keys()),
                    index=2,  # default 16:9 1920x1080
                    key="img2img_size",
                )
                params["size"] = _SIZE_PRESETS[size_label]
                params["aspect_ratio"] = None
                params["resolution"] = None

            # --- Seedream-specific params ---
            if workflow_source == "api" and workflow_key and "seedream" in workflow_key.lower():
                seed_val = st.number_input(
                    tr("img2img.seed"),
                    min_value=0,
                    max_value=9999999999,
                    value=0,
                    step=1,
                    key="img2img_seed",
                    help=tr("img2img.seed_help"),
                )
                params["seed"] = seed_val if seed_val > 0 else None

                style_val = st.text_input(
                    tr("img2img.style"),
                    placeholder=tr("img2img.style_placeholder"),
                    key="img2img_style",
                )
                if style_val.strip():
                    params["style"] = style_val.strip()

                quality_val = st.text_input(
                    tr("img2img.quality"),
                    placeholder=tr("img2img.quality_placeholder"),
                    key="img2img_quality",
                )
                if quality_val.strip():
                    params["quality"] = quality_val.strip()

        return params

    # ------------------------------------------------------------------
    # Right column – output preview
    # ------------------------------------------------------------------

    def _render_output(self, pixelle_video: Any, all_params: dict):
        with st.container(border=True):
            st.markdown(f"**{tr('img2img.section.output')}**")

            reference_image_path = all_params.get("reference_image_path")
            reference_image2_path = all_params.get("reference_image2_path")
            prompt = all_params.get("prompt", "")
            n_images = all_params.get("n_images", 1)
            workflow_key = all_params.get("workflow_key")
            workflow_source = all_params.get("workflow_source", "runninghub")
            size = all_params.get("size", "1920*1080")
            negative_prompt = all_params.get("negative_prompt")
            strength = all_params.get("strength", 0.7)
            seed = all_params.get("seed")
            style = all_params.get("style")
            quality = all_params.get("quality")
            workflow_info = all_params.get("workflow_info")
            image_source = all_params.get("image_source", "upload")
            aspect_ratio = all_params.get("aspect_ratio")
            resolution = all_params.get("resolution")

            # Validate workflow
            if not workflow_key:
                st.info(tr("img2img.select_workflow_hint"))
                st.button(
                    tr("img2img.btn_generate"),
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="img2img_generate_disabled",
                )
                return

            # Validate image
            if not reference_image_path:
                st.info(tr("img2img.no_image_hint"))
                st.button(
                    tr("img2img.btn_generate"),
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="img2img_generate_disabled2",
                )
                return

            if st.button(
                tr("img2img.btn_generate"),
                type="primary",
                use_container_width=True,
                key="img2img_generate",
            ):
                self._do_generate(
                    pixelle_video=pixelle_video,
                    reference_image_path=reference_image_path,
                    reference_image2_path=reference_image2_path,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    n_images=n_images,
                    workflow_key=workflow_key,
                    workflow_source=workflow_source,
                    size=size,
                    strength=strength,
                    workflow_info=workflow_info,
                    seed=seed,
                    style=style,
                    quality=quality,
                    image_source=image_source,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                )

        # Show previous results from session state
        if "img2img_results" in st.session_state and st.session_state["img2img_results"]:
            self._render_image_grid(st.session_state["img2img_results"])

    # ------------------------------------------------------------------
    # Generation logic
    # ------------------------------------------------------------------

    def _do_generate(
        self,
        pixelle_video: Any,
        reference_image_path: str,
        reference_image2_path: Optional[str] = None,
        prompt: str = "",
        negative_prompt: Optional[str] = None,
        n_images: int = 1,
        workflow_key: str = "",
        workflow_source: str = "runninghub",
        size: str = "1920*1080",
        strength: float = 0.7,
        workflow_info: Optional[dict] = None,
        seed: Optional[int] = None,
        style: Optional[str] = None,
        quality: Optional[str] = None,
        image_source: str = "upload",
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
    ):
        import random
        import json

        progress = st.progress(0)
        status = st.empty()
        start = time.time()

        # Resolve reference image to local path (download if URL)
        resolved_image_path: Optional[str] = None
        resolved_image2_path: Optional[str] = None

        def _download_image(url: str, prefix: str) -> str:
            """Download an image URL to a temp file and return the local path."""
            import httpx
            timeout = httpx.Timeout(60.0)
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                session_id = str(uuid.uuid4()).replace("-", "")[:12]
                temp_dir = Path(f"temp/img2img_{prefix}_{session_id}")
                temp_dir.mkdir(parents=True, exist_ok=True)
                ext = ".png"
                content_type = resp.headers.get("content-type", "")
                if "jpeg" in content_type or "jpg" in content_type:
                    ext = ".jpg"
                elif "webp" in content_type:
                    ext = ".webp"
                resolved_path = temp_dir / f"reference{ext}"
                with open(resolved_path, "wb") as f:
                    f.write(resp.content)
                return str(resolved_path.absolute())

        try:
            if reference_image_path:
                if reference_image_path.startswith("http"):
                    status.text(tr("img2img.downloading_image1"))
                    resolved_image_path = _download_image(reference_image_path, "ref")
                else:
                    resolved_image_path = reference_image_path

            if reference_image2_path:
                if reference_image2_path.startswith("http"):
                    status.text(tr("img2img.downloading_image2"))
                    resolved_image2_path = _download_image(reference_image2_path, "ref2")
                else:
                    resolved_image2_path = reference_image2_path
        except Exception as e:
            logger.error(f"Failed to resolve reference image: {e}")
            status.text("")
            progress.empty()
            st.error(tr("img2img.url_download_error"))
            return

        try:
            async def _generate():
                task_dir, _task_id = create_task_output_dir()
                save_dir = os.path.join(task_dir, "images")
                os.makedirs(save_dir, exist_ok=True)

                results: List[str] = []

                if is_api_workflow(workflow_key):
                    # --- API mode ---
                    api_media = getattr(pixelle_video, "api_media", None)
                    if not api_media:
                        raise RuntimeError("API media service not available")

                    # workflow_key like: "api/dashscope/wan2.7-image"
                    parts = workflow_key.split("/")
                    model_name = parts[2] if len(parts) > 2 else "wan2.7-image"
                    is_seedream = "seedream" in model_name.lower()

                    for i in range(n_images):
                        status.text(tr("img2img.generating", index=i + 1, total=n_images))
                        progress.progress(int((i / n_images) * 100))

                        # Build prompt with negative prompt
                        full_prompt = prompt or ""
                        if negative_prompt and full_prompt:
                            full_prompt = f"{full_prompt}. Avoid: {negative_prompt}"
                        elif negative_prompt:
                            full_prompt = f"Avoid: {negative_prompt}"

                        # Use the unified media service
                        media_params: Dict[str, Any] = {
                            "prompt": full_prompt,
                            "workflow": workflow_key,
                            "media_type": "image",
                            "image_paths": [resolved_image_path],
                        }

                        # Pass size as width/height
                        size_parts = size.split("*")
                        if len(size_parts) == 2:
                            media_params["width"] = int(size_parts[0])
                            media_params["height"] = int(size_parts[1])

                        # Seedream-specific kwargs
                        if is_seedream:
                            current_seed = seed if seed else random.randint(0, 9999999999)
                            media_params["seed"] = current_seed
                            if style:
                                media_params["style"] = style
                            if quality:
                                media_params["quality"] = quality
                        else:
                            # Pass strength for non-seedream models (DashScope edit)
                            media_params["strength"] = strength

                        media_result = await pixelle_video.media(**media_params)
                        if media_result and media_result.url:
                            results.append(media_result.url)

                else:
                    # --- RunningHub mode ---
                    kit = await pixelle_video._get_or_create_comfykit()
                    workflow_path = Path("workflows") / workflow_key

                    if not workflow_path.exists():
                        raise Exception(f"Workflow file does not exist: {workflow_path}")

                    with open(workflow_path, "r", encoding="utf-8") as f:
                        workflow_config = json.load(f)

                    if workflow_config.get("source") == "runninghub" and "workflow_id" in workflow_config:
                        workflow_input = workflow_config["workflow_id"]
                    else:
                        workflow_input = str(workflow_path)

                    full_prompt = prompt or ""
                    if negative_prompt and full_prompt:
                        full_prompt = f"{full_prompt}. Avoid: {negative_prompt}"
                    elif negative_prompt:
                        full_prompt = f"Avoid: {negative_prompt}"

                    # Check if this is a dual-image workflow
                    is_dual_image_wf = any(
                        kw in workflow_key.lower() for kw in _DUAL_IMAGE_WORKFLOW_KEYWORDS
                    )

                    if is_dual_image_wf:
                        # Dual-image workflow params (e.g. RH_RhartImageG2ImageToImage)
                        workflow_params: Dict[str, Any] = {
                            "prompt": full_prompt,
                            "image1": resolved_image_path,
                        }
                        if resolved_image2_path:
                            workflow_params["image2"] = resolved_image2_path
                        if aspect_ratio:
                            workflow_params["aspectRatio"] = aspect_ratio
                        if resolution:
                            workflow_params["resolution"] = resolution
                        if seed:
                            workflow_params["seed"] = seed
                    else:
                        workflow_params = {
                            "prompt": full_prompt,
                            "image": resolved_image_path,
                            "denoise": 1.0 - strength,  # strength→denoise mapping
                        }
                    param_mappings = workflow_config.get("param_mappings")

                    for i in range(n_images):
                        status.text(tr("img2img.generating", index=i + 1, total=n_images))
                        progress.progress(int((i / n_images) * 100))

                        result = await kit.execute(
                            workflow_input, workflow_params, param_mappings=param_mappings
                        )

                        # Extract image URL from result
                        generated_url: Optional[str] = None
                        if hasattr(result, "images") and result.images:
                            generated_url = result.images[0]
                        elif hasattr(result, "outputs") and result.outputs:
                            for _node_id, node_output in result.outputs.items():
                                if isinstance(node_output, dict) and "images" in node_output:
                                    images = node_output["images"]
                                    if images:
                                        generated_url = images[0]
                                        break
                                elif isinstance(node_output, dict):
                                    for _k, v in node_output.items():
                                        if isinstance(v, list):
                                            for item in v:
                                                if isinstance(item, dict) and ("filename" in item or "url" in item):
                                                    generated_url = item.get("url") or item.get("filename")
                                                    break
                                    if generated_url:
                                        break

                        if not generated_url:
                            continue

                        # Download to local
                        import httpx
                        timeout = httpx.Timeout(120.0)
                        async with httpx.AsyncClient(timeout=timeout) as client:
                            resp = await client.get(generated_url)
                            resp.raise_for_status()
                            local_path = os.path.join(save_dir, f"img2img_{i}.png")
                            with open(local_path, "wb") as f:
                                f.write(resp.content)
                            results.append(local_path)

                progress.progress(100)
                elapsed = time.time() - start
                status.text(tr("status.success"))

                # Store results in session state
                st.session_state["img2img_results"] = results
                st.session_state["img2img_elapsed"] = elapsed
                st.session_state["img2img_params"] = {
                    "prompt": prompt,
                    "n_images": str(n_images),
                    "workflow_key": workflow_key,
                    "size": size,
                    "strength": str(strength),
                    "seed": str(seed) if seed else "",
                }

                return results

            run_async(_generate())

            # Show success info
            results = st.session_state.get("img2img_results", [])
            elapsed = st.session_state.get("img2img_elapsed", 0)
            if results:
                st.success(
                    tr("img2img.generated_success", count=len(results), time=f"{elapsed:.1f}")
                )

        except Exception as e:
            logger.exception(e)
            status.text("")
            progress.empty()
            st.error(tr("status.error", error=str(e)))

    # ------------------------------------------------------------------
    # Image grid display
    # ------------------------------------------------------------------

    def _render_image_grid(self, results: List[str]):
        st.markdown("---")
        st.markdown(f"**{tr('img2img.results_title', count=len(results))}**")

        n = len(results)
        cols = st.columns(min(n, 2))

        for i, col in enumerate(cols):
            with col:
                if i < n:
                    path = results[i]
                    if os.path.exists(path):
                        st.image(path, use_container_width=True, caption=f"#{i + 1}")
                    elif path.startswith("http"):
                        st.image(path, use_container_width=True, caption=f"#{i + 1}")

                    # Download button
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            img_bytes = f.read()
                        fname = os.path.basename(path)
                        st.download_button(
                            label=tr("img2img.download"),
                            data=img_bytes,
                            file_name=fname,
                            mime="image/png",
                            use_container_width=True,
                            key=f"img2img_dl_{i}",
                        )

        # Batch download all as zip
        if n > 1:
            if st.button(tr("img2img.download_all"), key="img2img_download_all"):
                import zipfile
                import io

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for path in results:
                        if os.path.exists(path):
                            zf.write(path, os.path.basename(path))
                zip_buffer.seek(0)
                st.download_button(
                    label=tr("img2img.download_zip"),
                    data=zip_buffer.getvalue(),
                    file_name="img2img_images.zip",
                    mime="application/zip",
                    key="img2img_zip_dl",
                )

        # Regenerate button
        params = st.session_state.get("img2img_params", {})
        if params:
            if st.button(tr("img2img.regenerate"), key="img2img_regen", use_container_width=True):
                st.session_state.pop("img2img_results", None)
                st.rerun()


register_pipeline_ui(Img2ImgPipelineUI)
