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

目前已大致完成jieba，thulac，pkuseg，snownlp与hanlp的统一。

安装依赖
========
```bash
pip install jieba thulac pkuseg snownlp hanlp tensorflow
```

快速入门
========
```python
from interface import HanSeg
seg = HanSeg('jieba', user_dict="user_dict.txt")

print(seg.cut(["今天天气真好，适合出去散步。", "不要笑挑战么，有点意思。"]))
print(seg.pos("今天天气真好"))
print(seg.keywords("今天天气真好"))
print(seg.sentiment_analysis("今天天气真好"))
seg.add_word("word")
seg.del_word("word")
seg.cut_file(input_file, output_file)
seg.words_count(input_file, output_file)
```

* 用户词典与停用词文件格式说明：
    * 用户词典示例：每行格式为 词语 词性（可忽略）（如哈基米 n）。
    * 停用词文件示例：每行一个停用词，如的、了。

主要功能
========
* 标准分词 ✔️
* 带位置信息的分词 ✔️
* 词性标注 ✔️
* 关键词提取 ✔️
* 情感分析 ✔️ 使用SnowNLP（结果比较玄学，可能迁移到HanLP）
* 按用户配置对文件进行分词（不支持多进程） ✔️
* 快速切分文件 ✔️ 各引擎接口一致使用pkuseg快速切分文件接口，支持多进程
* 按停止词过滤输出 ✔️
* 拼音转换 ✔️
* 文件词频统计 ✔️
* 繁体转简体 ✔️ 使用SnowNLP
* 文本总结 ✔️ 使用SnowNLP
* 文本相似度 ✔️ 使用HanLP（结果比较玄学）
* 修改用户词典 ✔️
* 修改停止词字典 ❌
* 文本分类 ❌
* 词向量 ❌

## 各引擎对比

|特性	       |jieba	|thulac	|pkuseg	  |snownlp  |HanLP    |
|-------------|-------|---------|---------|---------|---------|
|核心优势|速度快、社区活跃、易用性强|高准确性|多领域预训练模型|功能丰富的NLP工具包|多模型支持、工业级性能、功能最全面|
|分词速度|⚡⚡⚡⚡|⚡⚡|⚡⚡⚡|⚡⚡⚡|⚡⚡⚡⚡|
|分词准确性|中等|高|非常高|中等|非常高|
|词性标注|✔️ |✔️ |✔️ |✔️ |✔️ |
|自定义词典支持|✔️ 动态修改|✔️ 需重载模型|✔️ 需重载模型|❌|✔️ 需重载模型|
|关键词提取|✔️ (TF-IDF/TextRank)|❌ (依赖其他引擎代理)|❌ (依赖其他引擎代理)|✔️ (TF-IDF)|❌ (依赖其他引擎代理)|
|情感分析|❌ (需代理到snownlp)|❌ (需代理到snownlp)|❌ (需代理到snownlp)|✔️|❌ (需代理到snownlp)|
|内存占用|低|高|中等|中等|高|
|适用场景|通用文本、实时处理|学术研究、高精度分词需求|专业领域、文本处理|情感分析、文本增强（非专业分词）|高精度分词、工业级性能、多模型支持|

功能代理机制：需要开启multi_engines=True

模型选择建议
========
* 追求速度与易用性  → jieba
* 高精度学术研究    → thulac
* 专业领域文本处理  → pkuseg
* 需要兼顾精度与速度的工业场景 → HanLP

jieba引擎独有的suggest_freq功能，暂时无法在其他引擎基础上实现。

对于关键词提取功能，由于仅有jieba与snownlp拥有这一功能，因此在使用thu和pku引擎时，会先使用自身引擎对语句进行切分，然后根据用户的配置使用jieba的关键词提取功能，用户可以选择IDF方式或者是textrank方式来提取关键词。（如果可能，会尽量将此功能代理到hanlp上）

情感分析只有snownlp引擎支持，使用其他引擎时，会先使用自身的cut方法对语句进行切分，然后再通过snownlp来得到情感分析结果。（如果可能，会尽量将此功能代理到hanlp上）

快速切分文件功能虽然各个引擎都能调用，但是内部统一使用pkuseg的快速切分文件接口以及pkuseg的相关配置，因此使用前需要注意对pkuseg相关配置进行调整。

注意，切分文件时请确保文件内的同一句话内没有换行符，也即是说，一行内可以有多句完整的话，但请不要把一句话拆成多行。

snownlp虽然可以修改词典，但是不会影响其行为，因为其有固定的词典，不使用自定义的词典。

示例
========
[示例运行脚本](https://github.com/Fuxuanmylove/han-seg/blob/main/example.py)

[示例配置文件](https://github.com/Fuxuanmylove/han-seg/blob/main/config.yaml)

FAQ
========
* Q：为什么修改用户词典之后没有立即生效？A：需要调用他们的reload_engine()方法来重载用户词典。
* Q：为什么snownlp修改用户词典后没有效果？A：snownlp的词典是固定的，无法修改。
* Q：使用cut_file_fast方法时程序停不下来？A：确保主程序包裹在if \_\_name\_\_ == '\_\_main\_\_':中。
* Q：使用cut_file_fast方法怎么反而速度更慢了？A：Windows平台上创建多进程开销很大。因此在文件规模并非极其大时，建议使用普通的cut_file方法。
* Q：使用hanlp模型时报了奇怪的错误。A：确保tensorflow已经安装。

本项目旨在尽量统一各个库的接口并统一输出形式便于用户使用。如有建议请务必提出！