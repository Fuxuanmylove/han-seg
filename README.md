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

主要功能
========
* 如下：
    * 标准分词 cut
    * 词性标注 pos
    * 关键词提取 keywords
    * 情感分析 sentiment_analysis
    * 切分文件 cut_file
    * 文本分类 待实现
    * 文本总结 待实现
    * 拼音转换 待实现
    * 繁体转简体 待实现
    * 带位置信息的分词 待实现
    * 文本相似度 待实现

## 功能支持表

| 功能        | jieba | thulac  | pkuseg  | snownlp |
|-------------|-------|---------|---------|---------|
| 标准分词     | ✔️    | ✔️     | ✔️     | ✔️      |
| 建议频率     | ✔️    | ❌     | ❌     | ❌      |
| 词性标注     | ✔️    | ✔️     | ✔️     | ✔️      |
| 关键词提取   | ✔️    | ✔️*    | ✔️*     | ✔️     |
| 情感分析     | ✔️*   | ✔️*    | ✔️*    | ✔️      |
| 切分文件     | ✔️    | ✔️     | ✔️     | ✔️      |

*代表需启用多引擎模式

jieba引擎独有的suggest_freq功能，暂时无法在其他引擎基础上实现。

对于关键词提取功能，由于仅有jieba与snownlp拥有这一功能，因此在使用thu和pku引擎时，会先使用自身引擎对语句进行切分，然后根据用户的配置使用jieba的关键词提取功能，用户可以选择IDF方式或者是textrank方式来提取关键词。

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
    
    print("增加单词")
    # seg1.add_word("紫色心情") # jieba 的add_word调用的是jieba.add_word，不会作用在user_dict上。
    seg2.add_word("紫色心情")
    seg3.add_word("紫色心情")

    print("删除单词")
    # seg1.del_word("紫色心情") # jieba 的del_word调用的是jieba.del_word，不会作用在user_dict上。
    seg2.del_word("紫色心情")
    seg3.del_word("紫色心情")
    
    # SnowNLP不支持增加或者删除单词

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
