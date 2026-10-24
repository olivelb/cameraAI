"""Optional Gradio frontend. Native Windows UI: python desktop.py."""

import logging
import os
import socket
import time

import gradio as gr

import camera_core as core
from camera_core import (
    cfg, ai, ui_log, get_logs_and_status, generate_mjpeg,
    save_email_settings, send_test_email, toggle_email_state,
    send_ws_command, set_resolution, change_model_logic, update_sam3_prompt_logic,
    update_conf, toggle_tracking, change_tracker, toggle_face_recognition,
    add_to_gallery, rebuild_gallery, start_learning, update_sensitivity,
    toggle_animal_recognition, add_animal_to_gallery, rebuild_animal_gallery,
    start_animal_learning, update_animal_sensitivity, ask_vlm,
    start_background_threads,
)
from platform_utils import apply_windows_boost, cleanup_port
from streaming import attach_stream

log = logging.getLogger(__name__)


def switch_source(mode_label: str):
    """Switch camera source (currently locked to Pi)."""
    core.cam_source_mode = "pi"
    ui_log("📷 Source locked to Pi")
    return gr.update(visible=True)


# ── Gradio UI Layout ────────────────────────────────────────────────────────

with gr.Blocks(title="AI Vision") as demo:
    gr.Markdown("# 👁️ Smart Vision")

    with gr.Row():
        with gr.Column():
            gr.HTML(
                '<div style="width:100%;text-align:center;">'
                '<img src="/stream" style="width:100%;max-width:100%;border-radius:8px;">'
                "</div>",
                label="Live AI Feed",
            )
            ai_status_output = gr.Textbox(
                label="AI Status", interactive=False, autoscroll=False
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Controls")

            source_radio = gr.Radio(
                ["🖥️ Pi Camera Server"],
                value="🖥️ Pi Camera Server",
                label="Source",
                visible=False,
            )

            with gr.Group(visible=True) as pi_group:
                with gr.Row():
                    res_dropdown = gr.Dropdown(
                        [f"{w}x{h}" for w, h in sorted(cfg.camera.stream_resolution_set())],
                        value="x".join(map(str, cfg.pi.default_resolution)),
                        label="Pi Resolution",
                    )
                    task_mode = gr.Radio(
                        ["Detection", "Segmentation"],
                        value="Detection",
                        label="Task Mode",
                    )
                fps_slider = gr.Slider(
                    cfg.pi.fps_min,
                    cfg.pi.fps_max,
                    value=cfg.pi.fps_default,
                    step=1,
                    label="FPS",
                )

            gr.Markdown("### 🧠 AI Engine")
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    [
                        "yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
                        "yolo26l.pt", "yolo26x.pt",
                        "yolo11n.pt", "yolo11s.pt", "yolov8n.pt",
                        "sam2_t.pt",
                        "sam3.pt",
                    ],
                    value=cfg.ai.default_model,
                    label="Model",
                    scale=3,
                )
                optimize_btn = gr.Button("⚡ Optimize (TRT)", scale=1)
            sam3_prompt = gr.Textbox(
                value="person, cat, dog",
                label="SAM3 concepts (comma-separated)",
                info="Used only by sam3.pt; analyzes every processed frame.",
            )

            gr.Markdown("### 🎯 Tracking")
            with gr.Row():
                tracking_toggle = gr.Checkbox(
                    label="Enable Tracking", value=False, scale=1
                )
                tracker_dropdown = gr.Dropdown(
                    ["BoT-SORT", "ByteTrack"],
                    value="BoT-SORT",
                    label="Tracker Algorithm",
                    scale=2,
                )

            with gr.Accordion("👤 Face Recognition", open=False):
                face_recog_toggle = gr.Checkbox(
                    label="Enable Face Recognition", value=False
                )
                gallery_status_box = gr.Textbox(
                    label="Gallery Status",
                    interactive=False,
                    lines=3,
                    value="Gallery empty — add photos and click Rebuild",
                )
                with gr.Row():
                    person_name_input = gr.Textbox(
                        label="Person Name", placeholder="e.g. Example Person", scale=2
                    )
                    face_photos_input = gr.File(
                        label="Photos",
                        file_count="multiple",
                        file_types=[".jpg", ".jpeg", ".png", ".webp"],
                        scale=3,
                    )
                with gr.Row():
                    add_person_btn = gr.Button("➕ Add to Gallery", scale=1)
                    rebuild_gallery_btn = gr.Button("🔄 Rebuild Gallery", scale=1)
                with gr.Row():
                    learn_btn = gr.Button(
                        "📸 Capture & Learn (Live)", variant="primary", scale=2
                    )
                    sensitivity_slider = gr.Slider(
                        0.2, 0.8,
                        value=cfg.ai.face_threshold,
                        step=0.01,
                        label="Sensitivity (Threshold)",
                    )

            with gr.Accordion("🐱 Animal Recognition", open=False):
                animal_recog_toggle = gr.Checkbox(
                    label="Enable Animal Recognition", value=False
                )
                animal_gallery_status_box = gr.Textbox(
                    label="Animal Gallery Status",
                    interactive=False,
                    lines=3,
                    value="Gallery empty — add photos and click Rebuild",
                )
                with gr.Row():
                    animal_name_input = gr.Textbox(
                        label="Pet Name", placeholder="e.g. Garfield", scale=2
                    )
                    animal_photos_input = gr.File(
                        label="Photos",
                        file_count="multiple",
                        file_types=[".jpg", ".jpeg", ".png", ".webp"],
                        scale=3,
                    )
                with gr.Row():
                    add_animal_btn = gr.Button("➕ Add to Gallery", scale=1)
                    rebuild_animal_gallery_btn = gr.Button(
                        "🔄 Rebuild Gallery", scale=1
                    )
                with gr.Row():
                    animal_learn_btn = gr.Button(
                        "📸 Capture & Learn (Live)", variant="primary", scale=2
                    )
                    animal_sensitivity_slider = gr.Slider(
                        0.6, 0.99,
                        value=cfg.ai.animal_threshold,
                        step=0.01,
                        label="Sensitivity (Threshold)",
                    )

            conf_slider = gr.Slider(0.1, 1.0, value=0.5, label="Confidence")

            with gr.Accordion("📧 Email Alerts", open=False):
                email_enable = gr.Checkbox(
                    label="Enable Alerts", value=core.email_config["enabled"]
                )
                email_sender = gr.Textbox(
                    label="Sender Email (Gmail)", value=core.email_config["sender"]
                )
                email_password = gr.Textbox(
                    label="App Password",
                    type="password",
                    placeholder="Leave empty to keep the stored password",
                )
                email_receiver = gr.Textbox(
                    label="Receiver Email", value=core.email_config["receiver"]
                )
                save_btn = gr.Button("Save Settings")
                email_status = gr.Textbox(label="Status", interactive=False)
                save_btn.click(
                    save_email_settings,
                    inputs=[email_enable, email_sender, email_password, email_receiver],
                    outputs=email_status,
                )
                test_btn = gr.Button("📨 Send Test Email")
                test_btn.click(send_test_email, outputs=email_status)

            gr.Markdown("### 📜 System Logs")
            log_output = gr.Textbox(label="Console Output", lines=5, interactive=False)
            log_timer = gr.Timer(cfg.ui.log_refresh_interval)
            log_timer.tick(
                get_logs_and_status, outputs=[log_output, ai_status_output]
            )

            gr.Markdown("### 💬 Vision Assistant")
            chat = gr.ChatInterface(fn=ask_vlm)

            info_output = gr.Textbox(label="System Status", interactive=False)

    # ── Event wiring ─────────────────────────────────────────────────────────
    source_radio.change(switch_source, inputs=source_radio, outputs=[pi_group])
    conf_slider.change(update_conf, inputs=conf_slider)
    res_dropdown.change(set_resolution, inputs=res_dropdown, outputs=info_output)
    fps_slider.release(
        lambda x: send_ws_command("fps", int(x)), inputs=fps_slider, outputs=info_output
    )
    email_enable.change(toggle_email_state, inputs=email_enable)
    model_dropdown.change(
        change_model_logic,
        inputs=[model_dropdown, task_mode, sam3_prompt],
        outputs=info_output,
    )
    task_mode.change(
        change_model_logic,
        inputs=[model_dropdown, task_mode, sam3_prompt],
        outputs=info_output,
    )
    sam3_prompt.change(update_sam3_prompt_logic, inputs=sam3_prompt, outputs=info_output)
    optimize_btn.click(ai.export_to_tensorrt, inputs=[], outputs=info_output)
    tracking_toggle.change(toggle_tracking, inputs=tracking_toggle)
    tracker_dropdown.change(change_tracker, inputs=tracker_dropdown)
    face_recog_toggle.change(
        toggle_face_recognition, inputs=face_recog_toggle, outputs=gallery_status_box
    )
    add_person_btn.click(
        add_to_gallery,
        inputs=[person_name_input, face_photos_input],
        outputs=gallery_status_box,
    )
    rebuild_gallery_btn.click(rebuild_gallery, outputs=gallery_status_box)
    learn_btn.click(
        start_learning, inputs=person_name_input, outputs=gallery_status_box
    )
    sensitivity_slider.change(update_sensitivity, inputs=sensitivity_slider)
    animal_recog_toggle.change(
        toggle_animal_recognition,
        inputs=animal_recog_toggle,
        outputs=animal_gallery_status_box,
    )
    add_animal_btn.click(
        add_animal_to_gallery,
        inputs=[animal_name_input, animal_photos_input],
        outputs=animal_gallery_status_box,
    )
    rebuild_animal_gallery_btn.click(
        rebuild_animal_gallery, outputs=animal_gallery_status_box
    )
    animal_learn_btn.click(
        start_animal_learning,
        inputs=animal_name_input,
        outputs=animal_gallery_status_box,
    )
    animal_sensitivity_slider.change(
        update_animal_sensitivity, inputs=animal_sensitivity_slider
    )


# ── Main entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    cleanup_port(cfg.ui.server_port)
    apply_windows_boost()
    start_background_threads()

    # Get local IP for display
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    # Check for SSL certs
    ssl_cert = "cert.pem" if os.path.exists("cert.pem") else None
    ssl_key = "key.pem" if os.path.exists("key.pem") else None
    ssl_kwargs = {}
    if ssl_cert and ssl_key:
        ssl_kwargs = {"ssl_certfile": ssl_cert, "ssl_keyfile": ssl_key}
    scheme = "https" if ssl_kwargs else "http"

    # Optional UI login — strongly recommended when exposing the app
    auth = None
    if cfg.ui.auth_username and cfg.ui.auth_password:
        auth = (cfg.ui.auth_username, cfg.ui.auth_password)
    elif cfg.ui.share:
        log.error(
            "ui.share is ENABLED but no auth_username/auth_password is "
            "configured — the app (live camera feed included) will be "
            "reachable by anyone holding the public link!"
        )

    print("\n" + "=" * 50)
    print("🚀  SERVER STARTING  🚀")
    print(f"📱  Local URL: {scheme}://localhost:{cfg.ui.server_port}")
    print(f"🌍  Network URL: {scheme}://{local_ip}:{cfg.ui.server_port}")
    print(f"🔐  Login: {'enabled' if auth else 'DISABLED'} | "
          f"Public share: {'ON' if cfg.ui.share else 'off'}")
    print("=" * 50 + "\n")

    app_server, _, _ = demo.launch(
        server_name=cfg.ui.server_host,
        server_port=cfg.ui.server_port,
        share=cfg.ui.share,
        auth=auth,
        theme=gr.themes.Base(),
        prevent_thread_lock=True,
        **ssl_kwargs,
    )

    # Attach MJPEG stream endpoint
    attach_stream(app_server, generate_mjpeg)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        core._STOP.set()
        demo.close()
