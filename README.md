han-seg
========

目的
========
* 集成以下（包括但不限于）多种主流中文分词库并统一标准接口：
    * jieba
    * thulac
    * pkuseg
    * snownlp
    * handlp

目前已大致完成jieba，thulac，pkuseg与snownlp的统一。

主要功能
========
* 如下：
    * 标准分词 ✔️
    * 带位置信息的分词 ✔️
    * 词性标注 ✔️
    * 关键词提取 ✔️
    * 情感分析 ✔️ 各引擎统一使用snownlp的情感分析接口
    * 按用户配置对文件进行分词（不支持多进程） ✔️
    * 快速切分文件 ✔️ 各引擎接口一致使用pkuseg快速切分文件接口，支持多进程
    * 按停止词过滤输出 ✔️
    * 拼音转换 ✔️
    * 文件词频统计 ✔️
    * 繁体转简体 ✔️ 使用SnowNLP
    * 文本总结 ✔️ 使用SnowNLP
    * 文本相似度 ✔️ 使用SnowNLP
    * 即时修改用户词典 ✔️ 需要注意，使用thulac和pkuseg时此操作可能会显著降低程序运行效率
    * 滞后修改用户词典 ❌
    * 修改停止词字典 ❌
    * 按词性过滤输出 ❌
    * 文本分类 ❌
    * 词向量 ❌
    * hanlp特有功能 ❌

## 功能支持表

| 功能        | jieba | thulac  | pkuseg  | snownlp |
|-------------|-------|---------|---------|---------|
| 标准分词     | ✔️    | ✔️     | ✔️     | ✔️      |
| 分词（带位置）     | ✔️    | ✔️     | ✔️     | ✔️      |
| 建议频率     | ✔️    | ❌     | ❌     | ❌      |
| 词性标注     | ✔️    | ✔️     | ✔️     | ✔️      |
| 关键词提取   | ✔️    | ✔️*    | ✔️*     | ✔️     |
| 情感分析     | ✔️*   | ✔️*    | ✔️*    | ✔️      |
| 切分文件     | ✔️    | ✔️     | ✔️     | ✔️      |
| 按停止词过滤输出     | ✔️    | ✔️     | ✔️     | ✔️      |
| 修改字典     | ✔️    | ✔️     | ✔️     | ✔️      |
| 文件词频统计     | ✔️    | ✔️     | ✔️     | ✔️      |

*代表需启用多引擎模式

jieba引擎独有的suggest_freq功能，暂时无法在其他引擎基础上实现。

对于关键词提取功能，由于仅有jieba与snownlp拥有这一功能，因此在使用thu和pku引擎时，会先使用自身引擎对语句进行切分，然后根据用户的配置使用jieba的关键词提取功能，用户可以选择IDF方式或者是textrank方式来提取关键词。

情感分析只有snownlp引擎支持，使用其他引擎时，会先使用自身的cut方法对语句进行切分，然后再通过snownlp来得到情感分析结果。

快速切分文件功能虽然各个引擎都能调用，但是内部统一使用pkuseg的快速切分文件接口以及pkuseg的相关配置，因此使用前需要注意对pkuseg相关配置进行调整。

注意，切分文件时请确保文件内的同一句话内没有换行符，也即是说，一行内可以有多句完整的话，但请不要把一句话拆成多行。

snownlp虽然可以修改词典，但是不会影响其行为，因为其有固定的词典，不使用自定义的词典。

下载
========
由于项目尚未完成，依赖尚不明确，建议下载项目源码压缩文件并在根目录创建所需文件。

示例
========
```python
from interface import HanSeg

USER_DICT = "user_data/user_dict.txt"
STOP_WORDS_PATH = "user_data/stop_words.txt" # If you set filt=False, you don't need to specify a stop words path.
CONFIG_PATH = "config.yaml"

def test():
    # 初始化thulac引擎
    seg1 = HanSeg('jieba', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    seg2 = HanSeg('thulac', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    seg3 = HanSeg('pkuseg', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    seg4 = HanSeg('snownlp', multi_engines=True, user_dict=USER_DICT, filt=True, stop_words_path=STOP_WORDS_PATH, config_path=CONFIG_PATH)
    text = "今天天气真好，适合出去散步。但是这并不代表我紫色心情不会开最大档。中国有句古话，识时务者为俊杰。"
    text2 = "不要笑挑战么，有点意思。"
    tradition = "「繁體字」「繁體中文」的叫法在臺灣亦很常見。"
    
    print("拼音") # 基于SnowNLP的实现
    print(HanSeg.pinyin(text))
    
    print("繁体转简体") # 基于SnowNLP的实现
    print(HanSeg.t2s(tradition))
    
    print("相似度") # 基于SnowNLP的实现
    print(HanSeg.similarity(text, text2))
    
    print("摘要") # 基于SnowNLP的实现
    print(HanSeg.summary(text, limit=2))

    seg1.suggest_freq(('今天', '天气'))

    print("分词")
    print(seg1.cut(text))
    print(seg2.cut(text))
    print(seg3.cut(text))
    print(seg4.cut(text))
    
    print(seg1.cut(text, with_position=True))
    print(seg2.cut(text, with_position=True))
    print(seg3.cut(text, with_position=True))
    print(seg4.cut(text, with_position=True))

    print("词性标注")
    print(seg1.pos(text))
    print(seg2.pos(text))
    print(seg3.pos(text))
    print(seg4.pos(text))

    print("关键词提取")
    print(seg1.keywords(text, limit=2))
    print(seg2.keywords(text, limit=2))
    print(seg3.keywords(text, limit=2))
    print(seg4.keywords(text, limit=2))

    print("情感分析")
    print(seg1.sentiment_analysis(text))
    print(seg2.sentiment_analysis(text))
    print(seg3.sentiment_analysis(text))
    print(seg4.sentiment_analysis(text))

    print("增加单词")
    # jieba 的add_word调用的是jieba.add_word，不会作用在user_dict上。
    seg1.add_word("哈基米", freq=100, flag='n')
    # 下面三个引擎中，传入freq将会被忽略。
    seg2.add_word("哈基米", flag='n')
    seg3.add_word("哈基米", flag='n')
    seg4.add_word("哈基米", flag='n')

    print("删除单词")
    # jieba 的del_word调用的是jieba.del_word，不会作用在user_dict上。
    seg1.del_word("哈基米")
    seg2.del_word("哈基米")
    seg3.del_word("哈基米")
    seg4.del_word("哈基米")
    
    # 虽然可以让SnowNLP操作用户词典，但是这种行为不会影响SnowNLP的行为与结果。

    print("切分文件") # 自定义切分文件，不支持多进程切分
    seg1.cut_file("user_data/input_file.txt", "user_data/output_file1.txt", batch_size=100)
    seg2.cut_file("user_data/input_file.txt", "user_data/output_file2.txt", batch_size=100)
    seg3.cut_file("user_data/input_file.txt", "user_data/output_file3.txt", batch_size=100)
    seg4.cut_file("user_data/input_file.txt", "user_data/output_file4.txt", batch_size=100)

    print("多进程切分文件") # 无论使用什么引擎，都会使用pkuseg的类方法进行切分，使用pkuseg的配置
    seg1.cut_file_fast("user_data/input_file.txt", "user_data/output_file_fast.txt", workers=10)
    
    print("词频统计")
    seg1.words_count("user_data/words_count_input.txt", "user_data/words_count_output1.txt")
    seg2.words_count("user_data/words_count_input.txt", "user_data/words_count_output2.txt")
    seg3.words_count("user_data/words_count_input.txt", "user_data/words_count_output3.txt")
    seg4.words_count("user_data/words_count_input.txt", "user_data/words_count_output4.txt")

    # 如果代码中含有cut_file_fast，务必以
    # if __name__ == '__main__':
    #     Your_Function()
    # 的形式运行脚本，否则会有意料不到的后果。
    # 这是由于此方法设计了多进程操作。

```

使用配置文件来控制引擎的工作方式
* config.yaml
```yaml
jieba:
  HMM: false
  tune: true
  withWeight: false
  allowPOS: "ns n vn v" # seperated by space
  dictionary: "" # empty string if not needed
  idf_path: ""
  keywords_method: "textrank" # textrank or tfidf
  cut_mode: "default" # default / full / search

thulac:
  model_path: ""
  t2s: false
  seg_only: false
  max_length: 50000
  keywords_method: "textrank" # textrank or tfidf
  idf_path: ""
  withWeight: false
  allowPOS: "ns n vn v" # seperated by space

pkuseg:
  model_name: "web" # news web medicine tourism default
  postag: true
  keywords_method: "textrank" # textrank or tfidf
  idf_path: ""
  withWeight: false
  allowPOS: "ns n vn v" # seperated by space
  verbose: false

snownlp:

```

本项目旨在尽量统一各个库的接口并统一输出形式便于用户使用。如有建议请务必提出！
