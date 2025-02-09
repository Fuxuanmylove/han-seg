han-seg
========

目的
========
* 集成以下多种主流中文分词库并统一标准接口：
    * jieba
    * thulac
    * pkuseg
    * snownlp
    * handlp

目前已大致完成jieba与thulac的统一，后续将继续完成剩下工具的统一。

下载
========
由于项目尚未完成，依赖尚不明确，建议下载项目源码压缩文件并在根目录创建所需文件。

示例
========
```python
from hanseg.interface import HanSeg
seg = HanSeg(engine='jieba', config_path='config.yaml') # Choose engine from jieba and thulac
text = "今天天气真好，适合出去散步。"
word = "花火"
seg.cut(text)
seg.pos(text)
seg.add_word(word) # if engine is jieba
seg.del_word(word) # if engine is jieba
seg.keyword_extract(text)
```

使用配置文件来控制引擎的工作方式
* config.yaml
```yaml
# You can modify the following configuration as needed, but don't delete any lines.
# For the file path, if you don't need to set it, just make it an empty string.

global:
  multi-engines: true # use other engines while using some method that is not supported by current engine

jieba:
  HMM: true
  filt: false
  tune: true
  topK: 20
  withWeight: false
  allowPOS: "ns n vn v" # seperated by space
  dictionary: ""
  user_dict: "./user_data/user_dict.txt" 
  stop_words: "./user_data/stop_words.txt"
  idf_path: ""
  keyword_extract_method: "textrank" # textrank or tfidf
  cut_mode: "default" # default / full / search

thulac:
  model_path: ""
  user_dict: "./user_data/user_dict.txt"
  stop_words: "./user_data/stop_words.txt"
  t2s: false
  seg_only: false
  filt: false
  topK: 20
  withWeight: false
  max_length: 50000
  keyword_extract_method: "textrank" # textrank or tfidf
  allowPOS: "ns n vn v" # seperated by space
  idf_path: ""
```

本项目为一个GitHub菜鸟所设计，旨在尽量统一各个库的接口并统一输出形式便于用户使用。如有建议请务必提出！