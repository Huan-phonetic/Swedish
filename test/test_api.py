from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

word = "hund"  # 你可以改成任意瑞典语单词
prompt = f"请用简明中文解释瑞典语单词「{word}」的词源，说明它和英语的关系、同源词、历史演变和现在的意思。如果有和英语的同源词，请举例说明。"

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 或 "gpt-4.1"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    print("GPT返回：")
    print(response.choices[0].message.content.strip())
except Exception as e:
    print(f"查询失败: {e}")
