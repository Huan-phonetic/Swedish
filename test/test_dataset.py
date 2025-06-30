from datasets import load_dataset, Audio
import soundfile as sf

print("loading...")
dataset = load_dataset('KBLab/rixvox-v2', split='train', streaming=True)
print("iterating...")
for i, item in enumerate(dataset):
    audio_array = item['audio']['array']
    sr = item['audio']['sampling_rate']
    print(i, audio_array.shape, sr)
    if i > 2:
        break
print("done")