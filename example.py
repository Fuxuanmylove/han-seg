from interface import HanSeg

# 初始化thulac引擎
seg1 = HanSeg(engine_name='jieba')
seg2 = HanSeg(engine_name='thulac')
seg3 = HanSeg(engine_name='pkuseg')
text = "今天天气真好，适合出去散步。"

# 分词
print(seg1.cut(text))
print(seg2.cut(text))
print(seg3.cut(text))

# 词性标注
print(seg1.pos(text))
print(seg2.pos(text))
print(seg3.pos(text))

# 关键词提取
print(seg1.keywords(text))
print(seg2.keywords(text))
print(seg3.keywords(text))