import gradio as gr
from datasets import load_dataset, Audio
import numpy as np
import random
import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import folium
import json

# ===================== 词频表加载 =====================
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

# ===================== API Key & 数据集 =====================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

dataset = load_dataset('KBLab/rixvox-v2', split='train', streaming=True)
dataset = dataset.cast_column('audio', Audio())

cached_items = []
CACHE_SIZE = 100

# ===================== 地图缓存功能 =====================
DISTRICT_CACHE_FILE = "data/district_coords.json"

if os.path.exists(DISTRICT_CACHE_FILE) and os.path.getsize(DISTRICT_CACHE_FILE) > 0:
    with open(DISTRICT_CACHE_FILE, 'r', encoding='utf-8') as f:
        DISTRICT_COORDS = json.load(f)
else:
    DISTRICT_COORDS = {}


def save_district_cache():
    with open(DISTRICT_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(DISTRICT_COORDS, f, ensure_ascii=False, indent=4)

def query_district_info_gpt(district):
    prompt = f"""请帮我查询瑞典历史选区或现存地区「{district}」的大致地理位置，并返回其经纬度和现代对应的瑞典地区（如省/市/县/kommun/län等），格式如下：

    瑞典地区: {district}

    经纬度: [纬度, 经度]
    现代地区: （现代瑞典的对应地区名称，尽量简明）

    只返回上面格式，确保数字是浮点数，纬度和经度之间用逗号分隔。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        lat, lon = 60.1282, 18.6435
        modern_region = district
        if "经纬度" in content:
            coord_line = content.split("经纬度:")[1].split("现代地区:")[0].strip()
            coord_line = coord_line.replace("[", "").replace("]", "").strip()
            parts = [x.strip() for x in coord_line.split(",") if x.strip()]
            if len(parts) == 2:
                try:
                    lat, lon = float(parts[0]), float(parts[1])
                except Exception:
                    lat, lon = 60.1282, 18.6435
        if "现代地区" in content:
            modern_region = content.split("现代地区:")[1].strip().split("\n")[0]
        return [lat, lon], modern_region
    except Exception as e:
        print(f"查询失败: {e}")
    return [60.1282, 18.6435], district  # 瑞典中心兜底

def get_district_info(district):
    if district in DISTRICT_COORDS:
        coords, modern_region = DISTRICT_COORDS[district]
        # 健壮性检查
        if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
            coords = [60.1282, 18.6435]
        return coords, modern_region
    else:
        coords, modern_region = query_district_info_gpt(district)
        # 再次健壮性检查
        if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
            coords = [60.1282, 18.6435]
        DISTRICT_COORDS[district] = [coords, modern_region]
        save_district_cache()
        return coords, modern_region

def render_map(district):
    coords, modern_region = get_district_info(district)
    m = folium.Map(location=coords, zoom_start=4.4, width=400, height=300)
    folium.Marker(location=coords, popup=modern_region).add_to(m)
    return m._repr_html_(), modern_region

# ===================== 词源缓存功能 =====================
ETYMOLOGY_CACHE_FILE = "data/etymology_cache.json"

if os.path.exists(ETYMOLOGY_CACHE_FILE):
    with open(ETYMOLOGY_CACHE_FILE, 'r', encoding='utf-8') as f:
        ETYMOLOGY_CACHE = json.load(f)
else:
    ETYMOLOGY_CACHE = {}

def save_etymology_cache():
    with open(ETYMOLOGY_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(ETYMOLOGY_CACHE, f, ensure_ascii=False, indent=4)

# ===================== 其他功能 =====================

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

class AudioState:
    def __init__(self, audio, sr):
        self.audio = audio
        self.sr = sr
        self.position = 0

state = {'audio_state': None, 'current_item': None}

def load_new_audio(autoplay=False):
    item = get_random_item()
    audio_array = item['audio']['array']
    sr = item['audio']['sampling_rate']
    state['audio_state'] = AudioState(audio_array, sr)
    state['current_item'] = item
    return update_ui()

def arrange_words(words, num_cols=8):
    rows = []
    for i in range(0, len(words), num_cols):
        row = words[i:i + num_cols]
        if len(row) < num_cols:
            row += [""] * (num_cols - len(row))
        rows.append(row)
    return pd.DataFrame(rows)

def update_ui():
    item = state['current_item']
    audio = item['audio']
    district = item.get('district', '')
    year = item.get('year', '')
    transcription = item.get('whisper_transcription', '')
    words = transcription.split()
    word_table = arrange_words(words, num_cols=8)
    audio_bytes = audio['array'].astype(np.float32)
    sr = audio['sampling_rate']
    map_html, modern_region = render_map(district)
    return (sr, audio_bytes), modern_region, year, word_table, "", map_html

def seek_audio(direction):
    audio_state = state['audio_state']
    if audio_state is None:
        return update_ui()
    pos = audio_state.position + direction
    pos = max(0, min(pos, len(audio_state.audio) / audio_state.sr - 1))
    audio_state.position = pos
    start = int(pos * audio_state.sr)
    audio_bytes = audio_state.audio[start:].astype(np.float32)
    item = state['current_item']
    transcription = item.get('whisper_transcription', '')
    words = transcription.split()
    word_table = arrange_words(words, num_cols=8)
    district = item.get('district', '')
    map_html, modern_region = render_map(district)
    return (audio_state.sr, audio_bytes), modern_region, item.get('year', ''), word_table, "", map_html



def stop_audio():
    audio_state = state['audio_state']
    if audio_state is None:
        return update_ui()
    audio_state.position = 0
    return update_ui()

def get_etymology(word, level):
    if word == "":
        return ""
    if (level == "初级" and word in TOP_100) or (level == "中级" and word in TOP_1000) or (level == "高级" and word in TOP_3000):
        return "该词为高频词，不查询词源。"
    # 查询缓存
    if word in ETYMOLOGY_CACHE:
        return ETYMOLOGY_CACHE[word]
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
        result = response.choices[0].message.content.strip()
        ETYMOLOGY_CACHE[word] = result
        save_etymology_cache()
        return result
    except Exception as e:
        return f"查询失败: {e}"


def on_word_click(evt: gr.SelectData, level):
    return get_etymology(evt.value, level)

# ===================== Gradio 页面 =====================

with gr.Blocks() as demo:
    gr.Markdown("# RixVox 数据集音频浏览器")

    level = gr.Radio(["初级", "中级", "高级"], value="初级", label="难度")

    with gr.Row():
        audio = gr.Audio(label="音频播放器", interactive=True)
        with gr.Column():
            district = gr.Textbox(label="District", interactive=False)
            year = gr.Textbox(label="Year", interactive=False)

    transcription = gr.Dataframe(label="Transcription (点击单词)", interactive=True, wrap=True)
    ety_output = gr.Textbox(label="词源解释", lines=4, interactive=False)
    map_html = gr.HTML(label="District 地图")

    with gr.Row():
        btn_prev = gr.Button("<< 向后1秒")
        btn_stop = gr.Button("■ 停止")
        btn_next = gr.Button("向前1秒 >>")
        btn_new = gr.Button("下一首（随机）")

    btn_new.click(load_new_audio, outputs=[audio, district, year, transcription, ety_output, map_html])
    btn_prev.click(lambda: seek_audio(-1), outputs=[audio, district, year, transcription, ety_output, map_html])
    btn_next.click(lambda: seek_audio(1), outputs=[audio, district, year, transcription, ety_output, map_html])
    btn_stop.click(stop_audio, outputs=[audio, district, year, transcription, ety_output, map_html])

    transcription.select(on_word_click, inputs=level, outputs=ety_output)

    demo.load(load_new_audio, outputs=[audio, district, year, transcription, ety_output, map_html])

print("before gradio launch")
demo.launch()
print("after gradio launch")