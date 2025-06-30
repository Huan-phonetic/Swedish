# Swedish Language Learning App

## 项目结构

- `data/`：存放数据集文件
- `scripts/`：数据处理和测试脚本
- `app/`：教学app主代码
- `requirements.txt`：依赖包列表

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 测试数据集加载：
   ```bash
   python scripts/test_dataset.py
   ``` 

点过了的单词不要重复查询，保存下来
按词频排序，简单的词就不要列出了。nltk的包里面有瑞典语包，已经有了，你去看看。
地点边上来个地图，打个点。
名字不要显示。