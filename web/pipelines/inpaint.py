"""
Inpaint Pipeline UI

Upload an image, paint a mask over the area to inpaint, then generate
using RunningHub or API mode.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from loguru import logger
from PIL import Image, ImageDraw

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

# Maximum image dimension for canvas display (keeps UI responsive)
_MAX_CANVAS_DIM = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_for_canvas(img: Image.Image) -> Image.Image:
    """Downscale an image so the longest side is at most _MAX_CANVAS_DIM."""
    w, h = img.size
    if max(w, h) <= _MAX_CANVAS_DIM:
        return img
    scale = _MAX_CANVAS_DIM / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _mask_from_canvas(json_data: Optional[dict], size: tuple) -> Optional[Image.Image]:
    """Render drawable-canvas JSON strokes into a white-on-black mask PIL Image.

    *json_data* is the ``json_data`` field returned by **st_canvas**.
    White pixels = painted area (inpaint target).

    Supports both legacy flat-coordinate paths and SVG-command-style paths.
    """
    if not json_data:
        return None

    strokes = json_data.get("strokes") or json_data.get("objects") or []
    if not strokes:
        return None

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    for stroke in strokes:
        path = stroke.get("path") or []
        width = int(stroke.get("strokeWidth", 20))

        if not path or len(path) < 2:
            continue

        # Detect path format: SVG-command style (list of lists like ["M",x,y]) vs flat list [x,y,x,y,...]
        is_svg_path = isinstance(path[0], list)

        if stroke.get("type") == "path":
            points: list = []

            if is_svg_path:
                # SVG command path: [["M", x, y], ["Q", cx, cy, x, y], ["L", x, y], ...]
                # Extract the endpoint of each command
                for cmd in path:
                    if not isinstance(cmd, list) or len(cmd) < 3:
                        continue
                    cmd_type = cmd[0]
                    if cmd_type in ("M", "L"):
                        # Last two values are x, y
                        points.append((cmd[-2], cmd[-1]))
                    elif cmd_type == "Q":
                        # Last two values are the curve endpoint
                        points.append((cmd[-2], cmd[-1]))
                    # Skip unknown commands
            else:
                # Legacy flat list: [x0, y0, x1, y1, …]
                points = [(path[i], path[i + 1]) for i in range(0, len(path) - 1, 2)]

            if len(points) >= 2:
                draw.line(points, fill=255, width=width, joint="curve")
            elif len(points) == 1:
                r = width // 2
                cx, cy = points[0]
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

        elif stroke.get("type") == "rect":
            left = stroke.get("left", 0)
            top = stroke.get("originY") or stroke.get("top", 0)
            width_r = stroke.get("width", 0)
            height_r = stroke.get("height", 0)
            draw.rectangle([left, top, left + width_r, top + height_r], fill=255)

    return mask


def _save_temp_image(img: Image.Image, prefix: str) -> str:
    """Save a PIL image to a temp directory and return its absolute path."""
    session_id = str(uuid.uuid4()).replace("-", "")[:12]
    temp_dir = Path(f"temp/inpaint_{prefix}_{session_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{prefix}.png"
    img.save(str(path), format="PNG")
    return str(path.absolute())


def _download_image(url: str, prefix: str) -> str:
    """Download an image URL to a temp file and return the local path."""
    import httpx
    timeout = httpx.Timeout(60.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url)
        resp.raise_for_status()
    session_id = str(uuid.uuid4()).replace("-", "")[:12]
    temp_dir = Path(f"temp/inpaint_{prefix}_{session_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    ext = ".png"
    content_type = resp.headers.get("content-type", "")
    if "jpeg" in content_type or "jpg" in content_type:
        ext = ".jpg"
    elif "webp" in content_type:
        ext = ".webp"
    resolved_path = temp_dir / f"{prefix}{ext}"
    with open(resolved_path, "wb") as f:
        f.write(resp.content)
    return str(resolved_path.absolute())


# ---------------------------------------------------------------------------
# Pipeline UI
# ---------------------------------------------------------------------------

class InpaintPipelineUI(PipelineUI):
    """UI for the Image Inpaint pipeline."""

    name = "inpaint"
    icon = "🎨"

    @property
    def display_name(self):  # type: ignore[override]
        return tr("pipeline.inpaint.name")

    @property
    def description(self):  # type: ignore[override]
        return tr("pipeline.inpaint.description")

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
    # Left column – image upload, canvas mask, prompt
    # ------------------------------------------------------------------

    def _render_input(self) -> dict:
        params: Dict[str, Any] = {}

        with st.container(border=True):
            st.markdown(f"**{tr('inpaint.section.input')}**")

            # --- Image source ---
            source_key = "inpaint_image_source"
            source_options = ["upload", "url"]
            if st.session_state.get(source_key) not in source_options:
                st.session_state.pop(source_key, None)

            image_source = st.radio(
                tr("inpaint.image_source"),
                source_options,
                format_func=lambda s: (
                    tr("inpaint.source_upload") if s == "upload"
                    else tr("inpaint.source_url")
                ),
                horizontal=True,
                key=source_key,
            )

            reference_image_path: Optional[str] = None
            original_image: Optional[Image.Image] = None

            if image_source == "upload":
                uploaded_file = st.file_uploader(
                    tr("inpaint.upload"),
                    type=["jpg", "jpeg", "png", "webp"],
                    accept_multiple_files=False,
                    help=tr("inpaint.upload_help"),
                    key="inpaint_upload",
                )
                if uploaded_file:
                    original_image = Image.open(uploaded_file).convert("RGB")
                    reference_image_path = _save_temp_image(original_image, "original")
            else:
                url_input = st.text_input(
                    tr("inpaint.url_input"),
                    placeholder=tr("inpaint.url_placeholder"),
                    key="inpaint_url",
                )
                if url_input.strip():
                    try:
                        reference_image_path = _download_image(url_input.strip(), "original")
                        original_image = Image.open(reference_image_path).convert("RGB")
                    except Exception as e:
                        st.error(tr("inpaint.url_download_error"))
                        logger.error(f"Failed to download image for inpaint: {e}")

            params["reference_image_path"] = reference_image_path
            params["original_image"] = original_image

            # --- Canvas mask painting ---
            if original_image:
                st.markdown(f"**{tr('inpaint.canvas_title')}**")

                # Brush size slider
                brush_size = st.slider(
                    tr("inpaint.brush_size"),
                    min_value=5,
                    max_value=100,
                    value=20,
                    step=5,
                    key="inpaint_brush_size",
                    help=tr("inpaint.brush_size_help"),
                )

                # Prepare canvas display image (resized for performance)
                canvas_image = _resize_for_canvas(original_image)
                canvas_w, canvas_h = canvas_image.size

                # Drawing mode selection
                drawing_mode = st.radio(
                    tr("inpaint.drawing_mode"),
                    ["freedraw", "transform", "line", "rect", "circle"],
                    format_func=lambda m: {
                        "freedraw": tr("inpaint.mode_draw"),
                        "transform": tr("inpaint.mode_transform"),
                        "line": tr("inpaint.mode_line"),
                        "rect": tr("inpaint.mode_rect"),
                        "circle": tr("inpaint.mode_circle"),
                    }.get(m, m),
                    horizontal=True,
                    key="inpaint_drawing_mode",
                )

                col_clear, col_eraser = st.columns(2)
                with col_clear:
                    if st.button(tr("inpaint.clear_mask"), key="inpaint_clear_mask"):
                        st.session_state.pop("inpaint_canvas", None)
                        st.rerun()
                with col_eraser:
                    use_eraser = st.checkbox(tr("inpaint.eraser"), key="inpaint_eraser")

                # Canvas
                try:
                    from streamlit_drawable_canvas import st_canvas

                    canvas_result = st_canvas(
                        fill_color="rgba(255, 255, 255, 0.4)" if not use_eraser else "rgba(0, 0, 0, 0)",
                        stroke_width=brush_size,
                        stroke_color="#FFFFFF" if not use_eraser else "#000000",
                        background_image=canvas_image,
                        update_streamlit=True,
                        height=canvas_h,
                        drawing_mode=drawing_mode if not use_eraser else "freedraw",
                        key="inpaint_canvas",
                    )

                    # Generate mask from canvas strokes
                    mask_image: Optional[Image.Image] = None
                    if canvas_result and canvas_result.json_data:
                        mask_image = _mask_from_canvas(canvas_result.json_data, (canvas_w, canvas_h))

                    if mask_image:
                        # Check if any white pixels exist (i.e., user has painted something)
                        import numpy as np
                        mask_arr = np.array(mask_image)
                        if mask_arr.max() > 0:
                            # Resize mask back to original image size
                            mask_image = mask_image.resize(original_image.size, Image.Resampling.LANCZOS)
                            mask_path = _save_temp_image(mask_image, "mask")
                            params["mask_path"] = mask_path
                            params["mask_image"] = mask_image

                            # Show mask preview
                            with st.expander(tr("inpaint.mask_preview"), expanded=False):
                                st.image(mask_image, use_container_width=True, caption=tr("inpaint.mask_caption"))
                        else:
                            st.info(tr("inpaint.no_mask_hint"))

                except Exception as e:
                    st.error(tr("inpaint.canvas_missing"))
                    st.markdown(
                        "```bash\npip install streamlit-drawable-canvas\n```"
                    )
                    st.error(f"实际错误: {type(e).__name__}: {e}")

            # --- Prompt ---
            prompt = st.text_area(
                tr("inpaint.prompt"),
                placeholder=tr("inpaint.prompt_placeholder"),
                height=150,
                help=tr("inpaint.prompt_help"),
                key="inpaint_prompt",
            )

            with st.expander(tr("inpaint.negative_prompt"), expanded=False):
                negative_prompt = st.text_area(
                    tr("inpaint.negative_prompt_label"),
                    placeholder="low quality, blurry, distorted, …",
                    height=80,
                    key="inpaint_negative_prompt",
                )

            n_images = st.number_input(
                tr("inpaint.count"),
                min_value=1,
                max_value=4,
                value=1,
                step=1,
                key="inpaint_count",
            )

            params["prompt"] = prompt
            params["negative_prompt"] = negative_prompt.strip() or None
            params["n_images"] = int(n_images)
            params["image_source"] = image_source

        return params

    # ------------------------------------------------------------------
    # Middle column – style config (RunningHub / API)
    # ------------------------------------------------------------------

    def _render_style_config(self, pixelle_video: Any) -> dict:
        params: Dict[str, Any] = {}

        with st.container(border=True):
            st.markdown(f"**{tr('inpaint.section.config')}**")

            # --- mode selection (selfhost / runninghub / api) ---
            source_options = ["selfhost", "runninghub", "api"]
            source_key = "inpaint_workflow_source"
            if st.session_state.get(source_key) not in source_options:
                st.session_state.pop(source_key, None)

            workflow_source = st.radio(
                tr("inpaint.generation_source"),
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
                    key_contains="inpaint",
                )
            elif workflow_source == "selfhost":
                workflows = list_local_media_workflows(
                    pixelle_video,
                    "image",
                    "selfhost",
                    key_contains="inpaint",
                )
            else:
                workflows = list_api_media_workflows(pixelle_video, "image")

            workflow_options = [wf["display_name"] for wf in workflows]
            workflow_keys = [wf["key"] for wf in workflows]

            workflow_display = st.selectbox(
                tr("inpaint.workflow_select"),
                workflow_options if workflow_options else [tr("inpaint.no_workflow")],
                index=0,
                key="inpaint_workflow_select",
                help=workflow_select_help(),
            )

            workflow_key: Optional[str] = None
            workflow_info: Optional[dict] = None
            if workflow_options:
                idx = workflow_options.index(workflow_display)
                workflow_key = workflow_keys[idx]
                workflow_info = workflows[idx]
            else:
                st.warning(tr("inpaint.no_workflow_warning"))

            params["workflow_key"] = workflow_key
            params["workflow_source"] = workflow_source
            params["workflow_info"] = workflow_info

            # --- Strength / denoise slider ---
            strength = st.slider(
                tr("inpaint.strength"),
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help=tr("inpaint.strength_help"),
                key="inpaint_strength",
            )
            params["strength"] = strength

        return params

    # ------------------------------------------------------------------
    # Right column – output preview
    # ------------------------------------------------------------------

    def _render_output(self, pixelle_video: Any, all_params: dict):
        with st.container(border=True):
            st.markdown(f"**{tr('inpaint.section.output')}**")

            reference_image_path = all_params.get("reference_image_path")
            mask_path = all_params.get("mask_path")
            prompt = all_params.get("prompt", "")
            n_images = all_params.get("n_images", 1)
            workflow_key = all_params.get("workflow_key")
            workflow_source = all_params.get("workflow_source", "runninghub")
            negative_prompt = all_params.get("negative_prompt")
            strength = all_params.get("strength", 0.8)

            # Validate workflow
            if not workflow_key:
                st.info(tr("inpaint.select_workflow_hint"))
                st.button(
                    tr("inpaint.btn_generate"),
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="inpaint_generate_disabled",
                )
                return

            # Validate image
            if not reference_image_path:
                st.info(tr("inpaint.no_image_hint"))
                st.button(
                    tr("inpaint.btn_generate"),
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="inpaint_generate_disabled2",
                )
                return

            # Validate mask
            if not mask_path:
                st.info(tr("inpaint.no_mask_hint"))
                st.button(
                    tr("inpaint.btn_generate"),
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="inpaint_generate_disabled3",
                )
                return

            if st.button(
                tr("inpaint.btn_generate"),
                type="primary",
                use_container_width=True,
                key="inpaint_generate",
            ):
                self._do_generate(
                    pixelle_video=pixelle_video,
                    reference_image_path=reference_image_path,
                    mask_path=mask_path,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    n_images=n_images,
                    workflow_key=workflow_key,
                    workflow_source=workflow_source,
                    strength=strength,
                )

        # Show previous results from session state
        if "inpaint_results" in st.session_state and st.session_state["inpaint_results"]:
            self._render_image_grid(st.session_state["inpaint_results"])

    # ------------------------------------------------------------------
    # Generation logic
    # ------------------------------------------------------------------

    def _do_generate(
        self,
        pixelle_video: Any,
        reference_image_path: str,
        mask_path: str,
        prompt: str = "",
        negative_prompt: Optional[str] = None,
        n_images: int = 1,
        workflow_key: str = "",
        workflow_source: str = "runninghub",
        strength: float = 0.8,
    ):
        progress = st.progress(0)
        status = st.empty()
        start = time.time()

        try:
            async def _generate():
                task_dir, _task_id = create_task_output_dir()
                save_dir = os.path.join(task_dir, "images")
                os.makedirs(save_dir, exist_ok=True)

                results: List[str] = []

                # Build prompt with negative prompt
                full_prompt = prompt or ""
                if negative_prompt and full_prompt:
                    full_prompt = f"{full_prompt}. Avoid: {negative_prompt}"
                elif negative_prompt:
                    full_prompt = f"Avoid: {negative_prompt}"

                if is_api_workflow(workflow_key):
                    # --- API mode ---
                    api_media = getattr(pixelle_video, "api_media", None)
                    if not api_media:
                        raise RuntimeError("API media service not available")

                    for i in range(n_images):
                        status.text(tr("inpaint.generating", index=i + 1, total=n_images))
                        progress.progress(int((i / n_images) * 100))

                        media_params: Dict[str, Any] = {
                            "prompt": full_prompt,
                            "workflow": workflow_key,
                            "media_type": "image",
                            "image_paths": [reference_image_path],
                            "mask_path": mask_path,
                            "strength": strength,
                        }

                        media_result = await pixelle_video.media(**media_params)
                        if media_result and media_result.url:
                            results.append(media_result.url)

                else:
                    # --- RunningHub mode ---
                    workflow_path = Path("workflows") / workflow_key

                    if not workflow_path.exists():
                        raise Exception(f"Workflow file does not exist: {workflow_path}")

                    with open(workflow_path, "r", encoding="utf-8") as f:
                        workflow_config = json.load(f)

                    source = workflow_config.get("source", "")
                    param_mappings = workflow_config.get("param_mappings")

                    # --- RunningHub V2 (AI App) ---
                    if source == "runninghub_v2" and "app_id" in workflow_config:
                        from pixelle_video.services.runninghub_v2_client import RunningHubV2Client

                        # Build v2 params (pass negative prompt separately)
                        v2_params: Dict[str, Any] = {
                            "prompt": prompt or "",
                            "negative_prompt": negative_prompt or "",
                            "image": reference_image_path,
                            "mask": mask_path,
                            "denoise": strength,
                        }

                        rh_config = pixelle_video.config.get("comfyui", {})
                        v2_client = RunningHubV2Client(
                            api_key=rh_config.get("runninghub_api_key"),
                            instance_type=rh_config.get("runninghub_instance_type") or "default",
                        )

                        for i in range(n_images):
                            status.text(tr("inpaint.generating", index=i + 1, total=n_images))
                            progress.progress(int((i / n_images) * 100))

                            image_urls = await v2_client.execute(
                                app_id=workflow_config["app_id"],
                                params=v2_params,
                                param_mappings=param_mappings,
                            )

                            for url in image_urls:
                                # Download to local
                                import httpx
                                timeout = httpx.Timeout(120.0)
                                async with httpx.AsyncClient(timeout=timeout) as http_client:
                                    resp = await http_client.get(url)
                                    resp.raise_for_status()
                                    idx = len(results)
                                    local_path = os.path.join(save_dir, f"inpaint_{idx}.png")
                                    with open(local_path, "wb") as f:
                                        f.write(resp.content)
                                    results.append(local_path)

                        await v2_client.close()

                    # --- RunningHub V1 (legacy comfykit) ---
                    else:
                        kit = await pixelle_video._get_or_create_comfykit()

                        if workflow_config.get("source") == "runninghub" and "workflow_id" in workflow_config:
                            workflow_input = workflow_config["workflow_id"]
                        else:
                            workflow_input = str(workflow_path)

                        workflow_params: Dict[str, Any] = {
                            "prompt": full_prompt,
                            "negative_prompt": negative_prompt or "",
                            "image": reference_image_path,
                            "mask": mask_path,
                            "denoise": strength,
                        }

                        for i in range(n_images):
                            status.text(tr("inpaint.generating", index=i + 1, total=n_images))
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
                                local_path = os.path.join(save_dir, f"inpaint_{i}.png")
                                with open(local_path, "wb") as f:
                                    f.write(resp.content)
                                results.append(local_path)

                progress.progress(100)
                elapsed = time.time() - start
                status.text(tr("status.success"))

                # Store results in session state
                st.session_state["inpaint_results"] = results
                st.session_state["inpaint_elapsed"] = elapsed
                st.session_state["inpaint_params"] = {
                    "prompt": prompt,
                    "n_images": str(n_images),
                    "workflow_key": workflow_key,
                    "strength": str(strength),
                }

                return results

            run_async(_generate())

            # Show success info
            results = st.session_state.get("inpaint_results", [])
            elapsed = st.session_state.get("inpaint_elapsed", 0)
            if results:
                st.success(
                    tr("inpaint.generated_success", count=len(results), time=f"{elapsed:.1f}")
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
        st.markdown(f"**{tr('inpaint.results_title', count=len(results))}**")

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
                            label=tr("inpaint.download"),
                            data=img_bytes,
                            file_name=fname,
                            mime="image/png",
                            use_container_width=True,
                            key=f"inpaint_dl_{i}",
                        )

        # Regenerate button
        params = st.session_state.get("inpaint_params", {})
        if params:
            if st.button(tr("inpaint.regenerate"), key="inpaint_regen", use_container_width=True):
                st.session_state.pop("inpaint_results", None)
                st.rerun()


register_pipeline_ui(InpaintPipelineUI)
