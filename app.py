"""Gradio entry point; import spaces before initializing Torch for ZeroGPU."""
import os
import spaces
import gradio as gr

from inference import InferenceEngine
from serving import prune_session_history

# ZeroGPU requires CUDA initialization outside the decorated request function.
engine = InferenceEngine(device=os.environ.get("BAL_DEVICE", "cuda" if os.environ.get("SPACE_ID") else "cpu"))


@spaces.GPU(duration=30)
def gpu_reply(message, history, response_length):
    return engine.reply(message, history, response_length)


def respond(message, history, response_length):
    try:
        response = gpu_reply(message, history, response_length)
        # Gradio's state lives in this process; the GPU worker gets a copy.
        prune_session_history(history)
        return response
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


demo = gr.ChatInterface(
    fn=respond,
    title="Balochi Writing Lab",
    description=(
        "Try the experimental Balochi model. Ask for a short answer and specify Arabic or Latin script. "
        "It can use the wrong script, repeat words or give incorrect answers; native-speaker review is pending. "
        "[Model and measured results](https://huggingface.co/shah-bakhsh/balcorpus-qwen3-0.6b-v1). "
        "Free GPU availability and usage limits apply."
    ),
    textbox=gr.Textbox(placeholder="Write a message or ask for a short Balochi phrase…", max_length=4000),
    additional_inputs=[gr.Slider(16, 128, value=64, step=1, label="Maximum response tokens")],
    examples=[["Write one short greeting in Balochi using Arabic script.", 64], ["Write one short greeting in Balochi using Latin script.", 64]],
    cache_examples=False,
    flagging_mode="never",
    analytics_enabled=False,
    save_history=False,
    concurrency_limit=1,
    api_name="chat",
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch()
