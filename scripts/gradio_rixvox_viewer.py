import gradio as gr
from datasets import load_dataset, Audio
import numpy as np
import random
import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件，获取 OpenAI API Key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 加载数据集（非 streaming）
dataset = load_dataset('KBLab/rixvox-v2', split='train', streaming=True)
dataset = dataset.cast_column('audio', Audio())

# 转为 list 以便随机访问（streaming 只能迭代，需先缓存一部分）
cached_items = []
CACHE_SIZE = 100  # 可调整缓存数量

ENABLE_OPENAI = os.getenv("ENABLE_OPENAI", "true").lower() == "true"

def fill_cache():
    if len(cached_items) < CACHE_SIZE:
        for item in dataset:
            cached_items.append(item)
            if len(cached_items) >= CACHE_SIZE:
                break

print("before fill_cache")
fill_cache()
print("after fill_cache")

def get_random_item():
    if not cached_items:
        fill_cache()
    return random.choice(cached_items)

# 音频播放器状态
class AudioState:
    def __init__(self, audio, sr):
        self.audio = audio
        self.sr = sr
        self.position = 0  # 当前播放位置（秒）
        self.stopped = False

state = {'audio_state': None, 'current_item': None}

def load_new_audio(autoplay=False):
    item = get_random_item()
    audio_array = item['audio']['array']
    if audio_array.dtype != 'int16':
        audio_array = (audio_array * 32767).astype('int16')
    sr = item['audio']['sampling_rate']
    state['audio_state'] = AudioState(audio_array, sr)
    state['current_item'] = item
    return update_ui(autoplay=autoplay)

def split_transcription(transcription):
    # 按空格分词，返回 HighlightedText 需要的格式
    words = transcription.split()
    return [(word, None) for word in words]

def update_ui(autoplay=False):
    item = state['current_item']
    audio = item['audio']
    # 资料
    name = item.get('name', '')
    party = item.get('party', '')
    role = item.get('role', '')
    district = item.get('district', '')
    year = item.get('year', '')
    transcription = item.get('whisper_transcription', '')
    transcription_words = split_transcription(transcription)
    # 音频数据
    audio_bytes = audio['array']
    sr = audio['sampling_rate']
    if autoplay:
        audio_value = (sr, audio_bytes, True)
    else:
        audio_value = (sr, audio_bytes)
    return (
        audio_value,
        name, party, role, district, year, transcription_words, ""
    )

def seek_audio(direction):
    # direction: +1 or -1 (秒)
    audio_state = state['audio_state']
    if audio_state is None:
        return update_ui()
    pos = audio_state.position + direction
    pos = max(0, min(pos, len(audio_state.audio) / audio_state.sr - 1))
    audio_state.position = pos
    # 取新片段
    start = int(pos * audio_state.sr)
    audio_bytes = audio_state.audio[start:]
    return (
        (audio_state.sr, audio_bytes),
        state['current_item'].get('name', ''),
        state['current_item'].get('party', ''),
        state['current_item'].get('role', ''),
        state['current_item'].get('district', ''),
        state['current_item'].get('year', ''),
        split_transcription(state['current_item'].get('whisper_transcription', '')),
        ""
    )

def stop_audio():
    audio_state = state['audio_state']
    if audio_state is None:
        return update_ui()
    audio_state.position = 0
    return update_ui()

# 词源查询

def get_etymology(word):
    if not ENABLE_OPENAI:
        return "（未启用 OpenAI，未实际查询，仅作测试）"
    prompt = (
        f"请用简明中文解释瑞典语单词「{word}」的词源，说明它和英语的关系、同源词、历史演变和现在的意思。如果有和英语的同源词，请举例说明。"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"查询失败: {e}"

def on_word_click(evt: gr.SelectData):
    yield "查询中，请稍候……"
    ety = get_etymology(evt.value)
    yield ety


with gr.Blocks() as demo:
    gr.Markdown("# RixVox 数据集音频浏览器")
    with gr.Row():
        audio = gr.Audio(label="音频播放器", interactive=True)
        with gr.Column():
            name = gr.Textbox(label="Name", interactive=False)
            party = gr.Textbox(label="Party", interactive=False)
            role = gr.Textbox(label="Role", interactive=False)
            district = gr.Textbox(label="District", interactive=False)
            year = gr.Textbox(label="Year", interactive=False)

    # 改成 Dataframe
    transcription = gr.Dataframe(label="Transcription (点击单词)", interactive=True, headers=["Word"])
    ety_output = gr.Textbox(label="词源解释", lines=4, interactive=False)

    with gr.Row():
        btn_prev = gr.Button("<< 向后1秒")
        btn_stop = gr.Button("■ 停止")
        btn_next = gr.Button("向前1秒 >>")
        btn_new = gr.Button("下一首（随机）")

    # 事件绑定
    btn_new.click(lambda: load_new_audio(autoplay=True), outputs=[audio, name, party, role, district, year, transcription, ety_output])
    btn_prev.click(lambda: seek_audio(-1), outputs=[audio, name, party, role, district, year, transcription, ety_output])
    btn_next.click(lambda: seek_audio(1), outputs=[audio, name, party, role, district, year, transcription, ety_output])
    btn_stop.click(stop_audio, outputs=[audio, name, party, role, district, year, transcription, ety_output])

    # 点击单词查词源，支持loading提示
    transcription.select(on_word_click, outputs=ety_output, queue=True)

    # 初始化
    demo.load(lambda: load_new_audio(autoplay=False), outputs=[audio, name, party, role, district, year, transcription, ety_output])


print("before gradio launch")
demo.launch()
print("after gradio launch") 