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

目前已大致完成jieba，thulac，pkuseg与snownlp的统一。

下载
========
由于项目尚未完成，依赖尚不明确，建议下载项目源码压缩文件并在根目录创建所需文件。

示例
========
```python
from interface import HanSeg

CONFIG_PATH = "config.yaml"

def test():
    # 初始化thulac引擎
    seg1 = HanSeg('jieba', CONFIG_PATH)
    seg2 = HanSeg('thulac', CONFIG_PATH)
    seg3 = HanSeg('pkuseg', CONFIG_PATH)
    seg4 = HanSeg('snownlp', CONFIG_PATH)
    text = "今天天气真好，适合出去散步。如果花火小姐是我的老婆，那么我将十分富有，这样我就再也不用打工了。想到这就觉得很开心！"

    seg1.suggest_freq(('今天', '天气'))

    print("分词")
    print(seg1.cut(text))
    print(seg2.cut(text))
    print(seg3.cut(text))
    print(seg4.cut(text))

    print("词性标注")
    print(seg1.pos(text))
    print(seg2.pos(text))
    print(seg3.pos(text))
    print(seg4.pos(text))

    print("关键词提取")
    print(seg1.keywords(text))
    print(seg2.keywords(text))
    print(seg3.keywords(text))
    print(seg4.keywords(text))

    print("情感分析")
    print(seg1.sentiment_analysis(text))
    print(seg2.sentiment_analysis(text))
    print(seg3.sentiment_analysis(text))
    print(seg4.sentiment_analysis(text))

    print("切分文件")
    seg1.cut_file("input_file.txt", "output_file.txt")
    seg2.cut_file("input_file.txt", "output_file.txt")
    seg3.cut_file("input_file.txt", "output_file.txt")
    seg4.cut_file("input_file.txt", "output_file.txt")
    
if __name__ == '__main__':
    test()
```

使用配置文件来控制引擎的工作方式
* config.yaml
```yaml
log:
   ...

global:
   ...

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
