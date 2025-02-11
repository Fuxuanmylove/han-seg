# example.py

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

    print("切分文件") # 自定义切分文件，不支持多进程切分
    seg1.cut_file("input_file.txt", "output_file.txt")
    seg2.cut_file("input_file.txt", "output_file.txt")
    seg3.cut_file("input_file.txt", "output_file.txt")
    seg4.cut_file("input_file.txt", "output_file.txt")
    
    print("多进程切分文件") # 无论使用什么引擎，都会使用pkuseg的类方法进行切分，使用pkuseg的配置
    seg1.cut_file_fast("input_file.txt", "output_file_fast.txt", workers=10)
    
    # 如果代码中含有cut_file_fast，务必以
    # if __name__ == '__main__':
    #     Your_Function()
    # 的形式运行脚本，否则会有意料不到的后果。
    # 这是由于此方法设计了多进程操作。
    
if __name__ == '__main__':
    test()