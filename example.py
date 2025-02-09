from han_seg.interface import HanSeg

# 初始化thulac引擎
seg1 = HanSeg(engine='jieba', config_path='config.yaml')
seg2 = HanSeg(engine='thulac', config_path='config.yaml')
text = "今天天气真好，适合出去散步。"

# 分词
print(seg1.cut(text))
print(seg2.cut(text))

# 词性标注
print(seg1.pos(text))
print(seg2.pos(text))

# 关键词提取
print(seg1.keyword_extract(text))
print(seg2.keyword_extract(text))