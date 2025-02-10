import logging
from typing import List, Tuple, Union
import jieba
from jieba import posseg as pseg
from jieba import analyse
from base import HanSegBase, HanSegError
from snownlp import SnowNLP


class HanSegJieba(HanSegBase):
    """Implementation based on jieba."""
    def __init__(self, global_config: dict = {}, local_config: dict = {}):
        
        super().__init__(global_config, local_config)
        self.HMM = local_config.get('HMM', True)
        self.tune = local_config.get('tune', True)
        self.dictionary_path = local_config.get('dictionary', None)
        if self.dictionary_path:
            jieba.set_dictionary(self.dictionary_path)

        jieba.load_userdict(self.user_dict_path)

        self.cut_mode = local_config.get('cut_mode', 'default').lower()
        if self.cut_mode not in ('default', 'full', 'search'):
            raise HanSegError(f"Invalid cut mode: {self.cut_mode}.\nYou must set cut_mode to 'default', 'full' or 'search' in your config.")

    def cut(self, text: str) -> List[str]:
        if self.cut_mode == 'default':
            words = jieba.cut(text, HMM=self.HMM)
        elif self.cut_mode == 'full':
            words = jieba.cut(text, cut_all=True, HMM=self.HMM)
        elif self.cut_mode == 'search':
            words = jieba.cut_for_search(text, HMM=self.HMM)

        if self.filt:
            return [word for word in words if word not in self.stop_words]
        return list(words)
        
    def pos(self, text: str) -> List[Tuple[str, str]]:
        if self.filt:
            return [(word, pos) for word, pos in pseg.cut(text, HMM=self.HMM) if word not in self.stop_words]
        return [(word, pos) for word, pos in pseg.lcut(text, HMM=self.HMM)]
        
    def add_word(self, word: str, freq: int = 1, flag: str = None) -> None:
        jieba.add_word(word, freq, flag)
        
    def del_word(self, word: str) -> None:
        jieba.del_word(word)
        
    def suggest_freq(self, words) -> None:
        jieba.suggest_freq(words, tune=self.tune)
        
    def keywords(self, text: str) -> Union[List[str], List[Tuple[str, float]]]:
        if self.keywords_method == 'tfidf':
            return analyse.extract_tags(text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
        elif self.keywords_method == 'textrank':
            return analyse.textrank(text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)

    def sentiment_analysis(self, text: str) -> float:
        super().sentiment_analysis(text)
    
    def _process_chunk(self, lines: List[str]) -> List[str]:
        return [' '.join(self.cut(line)) + '\n' for line in lines]