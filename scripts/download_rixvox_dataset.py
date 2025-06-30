import os
os.environ["HF_HOME"] = "E:/Swedish"
os.environ["HF_DATASETS_CACHE"] = "E:/Swedish/datasets"
from datasets import load_dataset, Audio

# 目标保存目录
save_dir = r'E:\Swedish'
os.makedirs(save_dir, exist_ok=True)

# 加载数据集（非 streaming）
dataset = load_dataset('KBLab/rixvox-v2', split='train')

dataset = dataset.cast_column('audio', Audio())

for item in dataset:
    audio = item['audio']
    audio_bytes = audio['array']
    sampling_rate = audio['sampling_rate']
    audio_path = audio['path']
    # 用 id 作为文件名，保留原扩展名
    file_id = item['id']
    ext = os.path.splitext(audio_path)[-1] if audio_path else '.wav'
    out_path = os.path.join(save_dir, f'{file_id}{ext}')
    # 保存音频文件
    with open(out_path, 'wb') as f_out:
        f_out.write(audio['bytes'])
    print(f'Saved: {out_path}')

print('All audio files have been downloaded and saved.') 