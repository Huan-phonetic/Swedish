import gradio as gr
from datasets import load_dataset, Audio
import numpy as np
import random
import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

# 加载词频表

def load_wordlist(path, n):
    words = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            word = line.strip().split()[0]
            words.append(word)
            if len(words) >= n:
                break
    return set(words)

TOP_100 = load_wordlist('data/sv_3k.txt', 100)
TOP_1000 = load_wordlist('data/sv_3k.txt', 1000)
TOP_3000 = load_wordlist('data/sv_3k.txt', 3000)

# 加载 API Key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 加载数据集
dataset = load_dataset('KBLab/rixvox-v2', split='train', streaming=True)
dataset = dataset.cast_column('audio', Audio())

# 缓存数据
cached_items = []
CACHE_SIZE = 100

def fill_cache():
    if len(cached_items) < CACHE_SIZE:
        for item in dataset:
            cached_items.append(item)
            if len(cached_items) >= CACHE_SIZE:
                break

if not cached_items:
    fill_cache()

def get_random_item():
    if not cached_items:
        fill_cache()
    return random.choice(cached_items)

# 音频状态
class AudioState:
    def __init__(self, audio, sr):
        self.audio = audio
        self.sr = sr
        self.position = 0

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

def arrange_words(words, num_cols=8):
    rows = []
    for i in range(0, len(words), num_cols):
        row = words[i:i + num_cols]
        # 如果不够一行，用空字符串补齐
        if len(row) < num_cols:
            row += [""] * (num_cols - len(row))
        rows.append(row)
    return pd.DataFrame(rows)

def update_ui(autoplay=False):
    item = state['current_item']
    audio = item['audio']
    name = item.get('name', '')
    party = item.get('party', '')
    role = item.get('role', '')
    district = item.get('district', '')
    year = item.get('year', '')
    transcription = item.get('whisper_transcription', '')
    words = transcription.split()
    word_table = arrange_words(words, num_cols=8)  # 这里设置每行单词数

    audio_bytes = audio['array']
    sr = audio['sampling_rate']
    if autoplay:
        audio_value = (sr, audio_bytes, True)
    else:
        audio_value = (sr, audio_bytes)
    return audio_value, name, party, role, district, year, word_table, ""

def seek_audio(direction):
    audio_state = state['audio_state']
    if audio_state is None:
        return update_ui()
    pos = audio_state.position + direction
    pos = max(0, min(pos, len(audio_state.audio) / audio_state.sr - 1))
    audio_state.position = pos
    start = int(pos * audio_state.sr)
    audio_bytes = audio_state.audio[start:]
    item = state['current_item']
    transcription = item.get('whisper_transcription', '')
    words = transcription.split()
    word_table = arrange_words(words, num_cols=8)
    return (audio_state.sr, audio_bytes), item.get('name', ''), item.get('party', ''), item.get('role', ''), item.get('district', ''), item.get('year', ''), word_table, ""

def stop_audio():
    audio_state = state['audio_state']
    if audio_state is None:
        return update_ui()
    audio_state.position = 0
    return update_ui()

def get_etymology(word, level):
    if word == "":
        return ""
    # 难度过滤
    if (level == "初级" and word in TOP_100) or \
       (level == "中级" and word in TOP_1000) or \
       (level == "高级" and word in TOP_3000):
        return "该词为高频词，不查询词源。"
    prompt = f"""请用简明中文解释瑞典语单词「{word}」的词源，内容包含：
    - 英文翻译
    - 英文或其他语言的同源词（如果没有，请明确写\"无\"）
    - 历史演变（词形变化、含义变化）
    - 当前释义

    请严格按照下面的输出格式，用中文回答：

    瑞典语单词: {word}

    英文翻译: （这里填英文翻译）

    词源: （这里详细描述词源，包括历史词形、含义变化、来源语言等）

    同源词: （列出英文或其他语言同源词，多个词用逗号分隔，如果没有请写\"无\"）

    当前含义: （这里写当前这个词的中文释义）
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"查询失败: {e}"

def on_word_click(evt: gr.SelectData, level):
    return get_etymology(evt.value, level)

with gr.Blocks() as demo:
    gr.Markdown("# RixVox 数据集音频浏览器")
    level = gr.Radio(["初级", "中级", "高级"], value="初级", label="难度")
    with gr.Row():
        audio = gr.Audio(label="音频播放器", interactive=True)
        with gr.Column():
            name = gr.Textbox(label="Name", interactive=False)
            party = gr.Textbox(label="Party", interactive=False)
            role = gr.Textbox(label="Role", interactive=False)
            district = gr.Textbox(label="District", interactive=False)
            year = gr.Textbox(label="Year", interactive=False)

    transcription = gr.Dataframe(label="Transcription (点击单词)", interactive=True, wrap=True)
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

    transcription.select(on_word_click, inputs=level, outputs=ety_output)

    demo.load(lambda: load_new_audio(autoplay=False), outputs=[audio, name, party, role, district, year, transcription, ety_output])

demo.launch()
