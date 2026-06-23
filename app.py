import gradio as gr

from src.front_utils import html_empty_box_for_output, root_css
from src.model import Model, MODEL_REGISTRY_ACL_SAN, MODEL_REGISTRY_ADCL
from src.session import create_session, cleanup_session
from src.tabs.classic import load_example_frames, load_example_audio, submit, update_threshold, apply_snr
from src.tabs.video import load_example_videos, organize_examples_for_gradio, submit_video, update_threshold_video
from src.tabs.comparison import submit_comparison, update_comparison_threshold, update_comparison_type

# TO TRY TO AVOID SERVER DISCONNECT ERRORS PRESUMABLY FROM TIMEOUTS
import httpx
import gradio.route_utils as ru  # or wherever gradio creates its client

original_client = httpx.AsyncClient

class TimeoutlessClient(httpx.AsyncClient):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("timeout", httpx.Timeout(None))  # no timeout
        super().__init__(*args, **kwargs)

httpx.AsyncClient = TimeoutlessClient

# ==================================== SIMPLE ENOUGH CALLBACKS =====================================

def update_versions(model_name: str):
    return gr.Dropdown(
        choices=CHOICES_VERSIONS[model_name],
        value=CHOICES_VERSIONS[model_name][0][1]
    )

def on_dropdown_change(thresh_choice: str, model_version: str):
    if hasattr(Model(model_version), thresh_choice):
        return gr.update(value=getattr(Model(model_version), thresh_choice))
    return gr.skip()

def on_shown_output(model_version):
    return gr.update(value=Model(model_version).univ_thresh, interactive=True), gr.update(interactive=True)

def update_run_btn(image_in, audio_in):
    if image_in is not None and audio_in is not None:
        return gr.update(interactive=True)
    return gr.update(interactive=False)

def update_run_btn_video(video_in):
    if video_in is not None:
        return gr.update(interactive=True)
    return gr.update(interactive=False)

def update_snr_button(audio_in):
    return update_run_btn_video(audio_in)

def load_audio_wrapper(audio_in, snr, state):
    state['original_audio'] = audio_in
    if audio_in == None:
        return gr.update(value='inf', interactive=False), state
    return gr.update(value='inf'), state

# ========================================== APPLICATION ===========================================

CHOICES_MODELS = ['ACL-SaN', 'ADCL']

# Build CHOICES_VERSIONS from the registry
CHOICES_VERSIONS = {}
for key, cfg in MODEL_REGISTRY_ACL_SAN.items():
    CHOICES_VERSIONS.setdefault(cfg['group'], []).append((cfg['display_name'], key))
for key, cfg in MODEL_REGISTRY_ADCL.items():
    CHOICES_VERSIONS.setdefault(cfg['group'], []).append((cfg['display_name'], key))

choices_models_init = CHOICES_MODELS[0]

title = "ACL-SaN and ADCL models demo"

with gr.Blocks(css=root_css, title=title) as demo:
    # Initialize session state per client
    session_state = gr.State(delete_callback=cleanup_session)
    demo.load(fn=create_session, outputs=session_state)

    with gr.Tabs():
        with gr.TabItem("Model comparisons"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        model_name_in_comp = gr.Dropdown(choices=CHOICES_MODELS, value=choices_models_init, label="Model")
                        output_type_toggle = gr.Dropdown(
                            choices=["Overlaid", "Segmentation Mask"],
                            value="Overlaid",
                            label="Output type"
                        )
                    with gr.Row():
                        image_in_comp = gr.Image(type='pil', label="Image Input", height=300)
                        image_in_comp.unrender()
                        with gr.Column(scale=1):
                            gr.Examples(
                                examples=load_example_frames(),
                                inputs=[image_in_comp],
                                label="Click to load an example frame",
                                examples_per_page=5,
                            )
                        with gr.Column(scale=10):
                            image_in_comp.render()
                    with gr.Row():
                        audio_in_comp = gr.Audio(label="Audio Input")
                        audio_in_comp.unrender()
                        with gr.Column(scale=1):
                            _example_pipe_comp = gr.State(None)
                            gr.Examples(
                                examples=load_example_audio(),
                                inputs=[audio_in_comp],
                                label="Click to load an audio",
                                examples_per_page=5,
                                outputs=[_example_pipe_comp],
                                fn=lambda audio: audio,
                                run_on_click=True,
                            )
                        with gr.Column(scale=10):
                            audio_in_comp.render()
                    snr_comp = gr.Radio(
                        choices=[('\u221e', 'inf'), ('20', '20'), ('10', '10'), ('5', '5')],
                        label="SNR Picker",
                        value='inf',
                        elem_classes="equal-radio",
                        interactive=False
                    )

                    snr_comp.change(
                        fn=apply_snr,
                        inputs=[audio_in_comp, snr_comp, session_state],
                        outputs=[audio_in_comp]
                    )

                with gr.Column(scale=3):
                    btn_comp = gr.Button("Run", interactive=False)
                    comp_html_out = gr.HTML(
                        value=html_empty_box_for_output,
                        padding=False,
                        elem_id='custom_table_comp',
                        container=True
                    )
                    with gr.Row():
                        threshold_slider_comp = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.01,
                            label="Threshold",
                            info="Default value is universal threshold",
                            interactive=False,
                            scale=4
                        )
                        dropdown_thresh_comp = gr.Dropdown(
                            choices=[('Universal Thresh.', 'univ_thresh'), ('Custom Thresh.', "custom")],
                            value="univ_thresh",
                            label="Choose Threshold",
                            interactive=False,
                            scale=1
                        )

                    threshold_slider_comp.change(
                        fn=update_comparison_threshold,
                        inputs=[output_type_toggle, dropdown_thresh_comp, threshold_slider_comp, session_state],
                        outputs=[comp_html_out],
                    )

                    threshold_slider_comp.release(
                        fn=lambda: 'custom',
                        outputs=[dropdown_thresh_comp],
                    )

                    dropdown_thresh_comp.change(
                        fn=update_comparison_threshold,
                        inputs=[output_type_toggle, dropdown_thresh_comp, threshold_slider_comp, session_state],
                        outputs=[comp_html_out],
                    )


            image_in_comp.change(
                fn=update_run_btn,
                inputs=[image_in_comp, audio_in_comp],
                outputs=btn_comp
            )
            audio_in_comp.change(
                fn=update_run_btn,
                inputs=[image_in_comp, audio_in_comp],
                outputs=btn_comp
            )

            audio_in_comp.change(
                fn=update_snr_button,
                inputs=[audio_in_comp],
                outputs=snr_comp
            )

            audio_in_comp.input(
                fn=load_audio_wrapper,
                inputs=[audio_in_comp, snr_comp, session_state],
                outputs=[snr_comp, session_state]
            )

            _example_pipe_comp.change(
                fn=load_audio_wrapper,
                inputs=[_example_pipe_comp, snr_comp, session_state],
                outputs=[snr_comp, session_state]
            )

            btn_comp.click(
                fn=submit_comparison,
                inputs=[image_in_comp, audio_in_comp, model_name_in_comp,
                        output_type_toggle, dropdown_thresh_comp, threshold_slider_comp, session_state],
                outputs=[comp_html_out, session_state]
            )
            output_type_toggle.change(
                fn=update_comparison_type,
                inputs=[image_in_comp, output_type_toggle, dropdown_thresh_comp, threshold_slider_comp, session_state],
                outputs=[comp_html_out, threshold_slider_comp, dropdown_thresh_comp]
            )

        # ============= IMAGE + AUDIO TAB =============
        with gr.TabItem("Classic"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_name_in = gr.Dropdown(choices=CHOICES_MODELS, value=choices_models_init, label="Model")
                        model_version_name_in = gr.Dropdown(
                            choices=CHOICES_VERSIONS[choices_models_init],
                            value=CHOICES_VERSIONS[choices_models_init][0][1],  # 'baseline'
                            label="Version"
                        )
                        model_name_in.change(fn=update_versions, inputs=model_name_in, outputs=model_version_name_in)
                    with gr.Row():
                        image_in = gr.Image(type='pil', label="Image Input", height=300)
                        image_in.unrender()
                        with gr.Column(scale=1):
                            gr.Examples(
                                examples=load_example_frames(),
                                inputs=[image_in],
                                label="Click to load an example frame",
                                examples_per_page=5,
                            )
                        with gr.Column(scale=10):
                            image_in.render()
                    with gr.Row():
                        audio_in = gr.Audio(label="Audio Input")
                        audio_in.unrender()
                        with gr.Column(scale=1):
                            _example_pipe = gr.State(None)
                            gr.Examples(
                                examples=load_example_audio(),
                                inputs=[audio_in],
                                label="Click to load an audio",
                                examples_per_page=5,
                                outputs=[_example_pipe],
                                fn=lambda audio: audio,
                                run_on_click=True,
                            )
                        with gr.Column(scale=10):
                            audio_in.render()
                    snr = gr.Radio(
                        choices=[('\u221e', 'inf'), ('20', '20'), ('10', '10'), ('5', '5')],
                        label="SNR Picker",
                        value='inf',
                        elem_classes="equal-radio",
                        interactive=False
                    )

                    snr.change(
                        fn=apply_snr,
                        inputs=[audio_in, snr, session_state],
                        outputs=[audio_in]
                    )

                with gr.Column():
                    btn = gr.Button("Run", interactive=False)
                    overlaid_out = gr.Image(type='pil', label="Overlaid with Original", height=300)
                    heatmap_out = gr.Image(type='pil', label="Heatmap (Grayscale)", height=300)
                    with gr.Row():
                        threshold_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.01,
                            label="Threshold",
                            info="Default value is universal threshold",
                            interactive=False,
                            scale=4
                        )
                        dropdown_thresh = gr.Dropdown(
                            choices=[('Universal Thresh.', 'univ_thresh'), ('Custom Thresh.', "custom")],
                            value="univ_thresh",
                            label="Choose Threshold",
                            interactive=False,
                            scale=1
                        )

                    threshold_slider.change(
                        fn=update_threshold,
                        inputs=[dropdown_thresh, threshold_slider, model_version_name_in, session_state],
                        outputs=heatmap_out,
                    )

                    threshold_slider.release(
                        fn=lambda: 'custom',
                        outputs=[dropdown_thresh],
                    )

                    dropdown_thresh.change(
                        fn=on_dropdown_change,
                        inputs=[dropdown_thresh, model_version_name_in],
                        outputs=threshold_slider
                    )

                    dropdown_thresh.change(
                        fn=update_threshold,
                        inputs=[dropdown_thresh, threshold_slider, model_version_name_in, session_state],
                        outputs=heatmap_out,
                    )

            image_in.change(
                fn=update_run_btn,
                inputs=[image_in, audio_in],
                outputs=btn
            )
            audio_in.change(
                fn=update_run_btn,
                inputs=[image_in, audio_in],
                outputs=btn
            )

            audio_in.change(
                fn=update_snr_button,
                inputs=[audio_in],
                outputs=snr
            )

            audio_in.input(
                fn=load_audio_wrapper,
                inputs=[audio_in, snr, session_state],
                outputs=[snr, session_state]
            )

            _example_pipe.change(
                fn=load_audio_wrapper,
                inputs=[_example_pipe, snr, session_state],
                outputs=[snr, session_state]
            )

            btn.click(
                fn=submit,
                inputs=[image_in, audio_in, model_name_in, model_version_name_in, threshold_slider, session_state],
                outputs=[heatmap_out, overlaid_out, session_state]
            ).then(
                fn=on_shown_output,
                inputs=model_version_name_in,
                outputs=[threshold_slider, dropdown_thresh]
            )

        # ============= VIDEO TAB =============
        with gr.TabItem("Video"):
            # Load example videos
            example_videos_dict = load_example_videos()
            example_videos_count = sum(len(v) for v in example_videos_dict.values())

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_name_in_video = gr.Dropdown(choices=CHOICES_MODELS, value=choices_models_init, label="Model")
                        model_version_name_in_video = gr.Dropdown(
                            choices=CHOICES_VERSIONS[choices_models_init],
                            value=CHOICES_VERSIONS[choices_models_init][0][1],  # 'baseline'
                            label="Version"
                        )
                        model_name_in_video.change(fn=update_versions,
                                                   inputs=model_name_in_video,
                                                   outputs=model_version_name_in_video)

                    video_in = gr.Video(label="Video Input", height=400)
                    # Display examples if available
                    if example_videos_count > 0:
                        gr.Markdown((
                            "**Available examples:**"
                            ' | '.join([f'{cat} ({len(vids)})' for cat, vids in example_videos_dict.items() if vids])
                        ))
                        example_videos = organize_examples_for_gradio(example_videos_dict)

                        if example_videos:
                            gr.Examples(
                                examples=example_videos,
                                inputs=[video_in],
                                label="Click to load an example video",
                                examples_per_page=5,
                            )
                with gr.Column():
                    btn_video = gr.Button("Run", interactive=False)
                    v_overlaid_out = gr.Video(label="Overlaid with Original", height=300)
                    v_heatmap_out = gr.Video(label="Heatmap (Grayscale)", height=300)
                    with gr.Row():
                        threshold_slider_video = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.01,
                            label="Threshold",
                            info="Default value is universal threshold",
                            interactive=False,
                            scale=4
                        )
                        dropdown_thresh_video = gr.Dropdown(
                            choices=[('Universal Thresh.', 'univ_thresh'), ('Custom Thresh.', "custom")],
                            value="univ_thresh",
                            label="Choose Threshold",
                            interactive=False,
                            scale=1
                        )

                    threshold_slider_video.change(
                        fn=update_threshold_video,
                        inputs=[dropdown_thresh_video, threshold_slider_video, model_version_name_in_video, session_state],
                        outputs=v_heatmap_out,
                    )

                    threshold_slider_video.release(
                        fn=lambda: 'custom',
                        outputs=[dropdown_thresh_video],
                    )

                    dropdown_thresh_video.change(
                        fn=on_dropdown_change,
                        inputs=[dropdown_thresh_video, model_version_name_in_video],
                        outputs=threshold_slider_video
                    )

                    dropdown_thresh_video.change(
                        fn=update_threshold_video,
                        inputs=[dropdown_thresh_video, threshold_slider_video, model_version_name_in_video, session_state],
                        outputs=v_heatmap_out,
                    )

            video_in.change(
                fn=update_run_btn_video,
                inputs=video_in,
                outputs=btn_video
            )

            btn_video.click(
                fn=submit_video,
                inputs=[video_in, model_name_in_video, model_version_name_in_video, threshold_slider_video, session_state],
                outputs=[v_heatmap_out, v_overlaid_out, session_state]
            ).then(
                fn=on_shown_output,
                inputs=model_version_name_in_video,
                outputs=[threshold_slider_video, dropdown_thresh_video]
            )

# init app
demo.queue(default_concurrency_limit=4)
demo.launch(server_name="0.0.0.0", server_port=7860)
