import gradio as gr

def on_word_click(evt: gr.SelectData):
    return f"你点击了：{evt.value}"

with gr.Blocks() as demo:
    words = [["hello"], ["world"], ["gradio"], ["python"]]
    table = gr.Dataframe(value=words, headers=["Word"], interactive=True)
    ety_output = gr.Textbox(label="词源解释", lines=4, interactive=False)

    table.select(on_word_click, outputs=ety_output)

demo.launch()
