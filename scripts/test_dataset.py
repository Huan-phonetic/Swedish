import sounddevice as sd
from datasets import load_dataset, Audio

# 加载数据集并解码音频字段
dataset = load_dataset("KBLab/rixvox-v2", streaming=True)
dataset["train"] = dataset["train"].cast_column("audio", Audio())
train_iter = iter(dataset["train"])

def play_next_sample():
    sample = next(train_iter)

    audio_array = sample["audio"]["array"]
    sample_rate = sample["audio"]["sampling_rate"]
    transcription = sample.get("whisper_transcription") or sample.get("text") or "(无转录)"
    speaker = sample.get("speaker_id", "N/A")

    print(f"Speaker: {speaker}")
    print(f"Transcription: {transcription}")

    print("Playing audio...")
    sd.play(audio_array, sample_rate)
    sd.wait()
    print("Playback finished.\n")

if __name__ == "__main__":
    # 连续播放前3条示范
    for _ in range(3):
        play_next_sample()

#根据地区、根据男女、基频、语速