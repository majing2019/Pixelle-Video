"""
Text-to-Image Pipeline UI

Generates images from text prompts, supporting RunningHub and API modes.
"""

import os
import time
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
# Size presets
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


class TextToImagePipelineUI(PipelineUI):
    """UI for the Text-to-Image pipeline."""

    name = "text_to_image"
    icon = "🖼️"

    @property
    def display_name(self):  # type: ignore[override]
        return tr("pipeline.text_to_image.name")

    @property
    def description(self):  # type: ignore[override]
        return tr("pipeline.text_to_image.description")

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
    # Left column – content input
    # ------------------------------------------------------------------

    def _render_input(self) -> dict:
        with st.container(border=True):
            st.markdown(f"**{tr('t2i.section.input')}**")

            prompt = st.text_area(
                tr("t2i.prompt"),
                placeholder=tr("t2i.prompt_placeholder"),
                height=200,
                help=tr("t2i.prompt_help"),
                key="t2i_prompt",
            )

            with st.expander(tr("t2i.negative_prompt"), expanded=False):
                negative_prompt = st.text_area(
                    tr("t2i.negative_prompt_label"),
                    placeholder="low quality, blurry, distorted, …",
                    height=80,
                    key="t2i_negative_prompt",
                )

            n_images = st.number_input(
                tr("t2i.count"),
                min_value=1,
                max_value=4,
                value=1,
                step=1,
                key="t2i_count",
            )

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt.strip() or None,
            "n_images": int(n_images),
        }

    # ------------------------------------------------------------------
    # Middle column – style config (RunningHub / API only)
    # ------------------------------------------------------------------

    def _render_style_config(self, pixelle_video: Any) -> dict:
        params: Dict[str, Any] = {}

        with st.container(border=True):
            st.markdown(f"**{tr('t2i.section.config')}**")

            # --- mode selection (runninghub / api) ---
            source_options = ["runninghub", "api"]
            source_key = "t2i_workflow_source"
            if st.session_state.get(source_key) not in source_options:
                st.session_state.pop(source_key, None)

            workflow_source = st.radio(
                tr("t2i.generation_source"),
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
                )
            else:
                workflows = list_api_media_workflows(pixelle_video, "image")

            workflow_options = [wf["display_name"] for wf in workflows]
            workflow_keys = [wf["key"] for wf in workflows]

            workflow_display = st.selectbox(
                tr("t2i.workflow_select"),
                workflow_options if workflow_options else [tr("t2i.no_workflow")],
                index=0,
                key="t2i_workflow_select",
                help=workflow_select_help(),
            )

            workflow_key: Optional[str] = None
            workflow_info: Optional[dict] = None
            if workflow_options:
                idx = workflow_options.index(workflow_display)
                workflow_key = workflow_keys[idx]
                workflow_info = workflows[idx]
            else:
                st.warning(tr("t2i.no_workflow_warning"))

            params["workflow_key"] = workflow_key
            params["workflow_source"] = workflow_source
            params["workflow_info"] = workflow_info

            # --- size presets ---
            size_label = st.selectbox(
                tr("t2i.size"),
                list(_SIZE_PRESETS.keys()),
                index=2,  # default 16:9 1920x1080
                key="t2i_size",
            )
            params["size"] = _SIZE_PRESETS[size_label]

            # --- Seedream-specific params ---
            if workflow_source == "api" and workflow_key and "seedream" in workflow_key.lower():
                seed_val = st.number_input(
                    tr("t2i.seed"),
                    min_value=0,
                    max_value=9999999999,
                    value=0,
                    step=1,
                    key="t2i_seed",
                    help=tr("t2i.seed_help"),
                )
                params["seed"] = seed_val if seed_val > 0 else None

                style_val = st.text_input(
                    tr("t2i.style"),
                    placeholder=tr("t2i.style_placeholder"),
                    key="t2i_style",
                )
                if style_val.strip():
                    params["style"] = style_val.strip()

                quality_val = st.text_input(
                    tr("t2i.quality"),
                    placeholder=tr("t2i.quality_placeholder"),
                    key="t2i_quality",
                )
                if quality_val.strip():
                    params["quality"] = quality_val.strip()

        return params

    # ------------------------------------------------------------------
    # Right column – output preview
    # ------------------------------------------------------------------

    def _render_output(self, pixelle_video: Any, all_params: dict):
        with st.container(border=True):
            st.markdown(f"**{tr('t2i.section.output')}**")

            prompt = all_params.get("prompt", "")
            n_images = all_params.get("n_images", 1)
            workflow_key = all_params.get("workflow_key")
            workflow_source = all_params.get("workflow_source", "runninghub")
            size = all_params.get("size", "1920*1080")
            negative_prompt = all_params.get("negative_prompt")
            seed = all_params.get("seed")
            style = all_params.get("style")
            quality = all_params.get("quality")
            workflow_info = all_params.get("workflow_info")

            # Validate
            if not workflow_key:
                st.info(tr("t2i.select_workflow_hint"))
                st.button(
                    tr("t2i.btn_generate"),
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="t2i_generate_disabled",
                )
                return

            if not prompt.strip():
                st.info(tr("t2i.enter_prompt_hint"))
                st.button(
                    tr("t2i.btn_generate"),
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="t2i_generate_disabled2",
                )
                return

            if st.button(tr("t2i.btn_generate"), type="primary", use_container_width=True, key="t2i_generate"):
                self._do_generate(
                    pixelle_video=pixelle_video,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    n_images=n_images,
                    workflow_key=workflow_key,
                    workflow_source=workflow_source,
                    size=size,
                    workflow_info=workflow_info,
                    seed=seed,
                    style=style,
                    quality=quality,
                )

        # Show previous results from session state
        if "t2i_results" in st.session_state and st.session_state["t2i_results"]:
            self._render_image_grid(st.session_state["t2i_results"])

    # ------------------------------------------------------------------
    # Generation logic
    # ------------------------------------------------------------------

    def _do_generate(
        self,
        pixelle_video: Any,
        prompt: str,
        negative_prompt: Optional[str],
        n_images: int,
        workflow_key: str,
        workflow_source: str,
        size: str,
        workflow_info: Optional[dict],
        seed: Optional[int],
        style: Optional[str],
        quality: Optional[str],
    ):
        import random
        import json

        progress = st.progress(0)
        status = st.empty()
        start = time.time()

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
                        status.text(tr("t2i.generating", index=i + 1, total=n_images))
                        progress.progress(int((i / n_images) * 100))

                        # Build prompt with negative prompt
                        full_prompt = prompt
                        if negative_prompt:
                            full_prompt = f"{prompt}. Avoid: {negative_prompt}"

                        # Use the unified media service
                        media_params: Dict[str, Any] = {
                            "prompt": full_prompt,
                            "workflow": workflow_key,
                            "media_type": "image",
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

                    full_prompt = prompt
                    if negative_prompt:
                        full_prompt = f"{prompt}. Avoid: {negative_prompt}"

                    workflow_params = {"prompt": full_prompt}
                    param_mappings = workflow_config.get("param_mappings")

                    for i in range(n_images):
                        status.text(tr("t2i.generating", index=i + 1, total=n_images))
                        progress.progress(int((i / n_images) * 100))

                        result = await kit.execute(workflow_input, workflow_params, param_mappings=param_mappings)

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
                            local_path = os.path.join(save_dir, f"t2i_{i}.png")
                            with open(local_path, "wb") as f:
                                f.write(resp.content)
                            results.append(local_path)

                progress.progress(100)
                elapsed = time.time() - start
                status.text(tr("status.success"))

                # Store results in session state
                st.session_state["t2i_results"] = results
                st.session_state["t2i_elapsed"] = elapsed
                st.session_state["t2i_params"] = {
                    "prompt": prompt,
                    "n_images": str(n_images),
                    "workflow_key": workflow_key,
                    "size": size,
                    "seed": str(seed) if seed else "",
                }

                return results

            run_async(_generate())

            # Show success info
            results = st.session_state.get("t2i_results", [])
            elapsed = st.session_state.get("t2i_elapsed", 0)
            if results:
                st.success(
                    tr("t2i.generated_success", count=len(results), time=f"{elapsed:.1f}")
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
        st.markdown(f"**{tr('t2i.results_title', count=len(results))}**")

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
                            label=tr("t2i.download"),
                            data=img_bytes,
                            file_name=fname,
                            mime="image/png",
                            use_container_width=True,
                            key=f"t2i_dl_{i}",
                        )

        # Batch download all as zip
        if n > 1:
            if st.button(tr("t2i.download_all"), key="t2i_download_all"):
                import zipfile
                import io

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for path in results:
                        if os.path.exists(path):
                            zf.write(path, os.path.basename(path))
                zip_buffer.seek(0)
                st.download_button(
                    label=tr("t2i.download_zip"),
                    data=zip_buffer.getvalue(),
                    file_name="t2i_images.zip",
                    mime="application/zip",
                    key="t2i_zip_dl",
                )

        # Regenerate with different seed
        params = st.session_state.get("t2i_params", {})
        if params:
            if st.button(tr("t2i.regenerate"), key="t2i_regen", use_container_width=True):
                # Clear results to trigger re-generation
                st.session_state.pop("t2i_results", None)
                st.rerun()


register_pipeline_ui(TextToImagePipelineUI)
