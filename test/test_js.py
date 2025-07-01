import gradio as gr

def debug_function(word):
    print(f"✅ 收到参数：{word}")
    return f"你点击了：{word}"

with gr.Blocks() as demo:
    gr.Markdown("## Gradio 5.35.0 点击测试")

    hidden_input = gr.Textbox(visible=False)  # 用来接收 js 参数
    result = gr.Textbox(label="点击结果")

    query_button = gr.Button("hej")

    query_button.click(
        debug_function,
        inputs=hidden_input,   # 必须绑定输入
        outputs=result,
        js="() => 'hej'"       # JS 传递参数
    )

demo.launch()
