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

目前已大致完成jieba，thulac与pkuseg的统一，后续将继续完成剩下工具的统一。

下载
========
由于项目尚未完成，依赖尚不明确，建议下载项目源码压缩文件并在根目录创建所需文件。

示例
========
```python
from hanseg.interface import HanSeg
seg = HanSeg(engine='jieba') # Choose engine from jieba and thulac
text = "今天天气真好，适合出去散步。"
word = "花火"
seg.cut(text)
seg.pos(text)
seg.keywords(text)
```

使用配置文件来控制引擎的工作方式
* config.yaml
```yaml
# You can modify the following configuration as needed, but don't delete any lines.
# For the file path, if you don't need to set it, just make it an empty string.

global:
  multi-engines: true # use other engines while using some method that is not supported by current engine

jieba:
   ...

thulac:
   ...

pkuseg:
   ...

snownlp:
   ...
```

本项目为一个GitHub菜鸟所设计，旨在尽量统一各个库的接口并统一输出形式便于用户使用。如有建议请务必提出！
