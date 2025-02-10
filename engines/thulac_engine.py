import logging
from typing import List, Tuple, Union
from base import HanSegBase, HanSegError
from jieba import analyse
from thulac import thulac
from snownlp import SnowNLP


class HanSegThulac(HanSegBase):
    """Implementation based on thulac."""

    def __init__(self, global_config: dict = {}, local_config: dict = {}):
        super().__init__(global_config, local_config)

        self.model_path = self.local_config.get('model_path', None)
        if not self.model_path:
            self.model_path = None
        self.seg_only = self.local_config.get('seg_only', False)
        self.t2s = self.local_config.get('t2s', False)
        self._thulac = thulac(model_path=self.model_path, seg_only=self.seg_only, T2S=self.t2s, user_dict=self.user_dict_path)
            
    def cut(self, text: str) -> List[str]:
        if self.filt:
            return [word[0] for word in self._thulac.cut(text) if word[0] not in self.stop_words]
        return [word[0] for word in self._thulac.cut(text)]
    
    def pos(self, text: str) -> List[Tuple[str, str]]:
        if self.seg_only:
            raise HanSegError("seg_only is true in config.")
        if self.filt:
            return [(word, pos) for word, pos in self._thulac.cut(text) if word not in self.stop_words]            
        return [(word, pos) for word, pos in self._thulac.cut(text)]
    
    def add_word(self, word: str, freq: int = 1, flag: str = None) -> None:
        if not self.user_dict_path:
            raise HanSegError("user_dict is not set in config.")
        
        with open(self.user_dict_path, 'a', encoding='utf-8') as f:
            line = word
            f.write(line.strip() + '\n')
            
        self._reload_engine()
        
    def del_word(self, word: str) -> None:
        if not self.user_dict_path:
            raise HanSegError("user_dict is not set in config.")

        with open(self.user_dict_path, 'r', encoding='utf-8') as f:
            lines = [l for l in f if l.split()[0] != word]

        with open(self.user_dict_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        self._reload_engine()

    def keywords(self, text: str) -> Union[List[str], List[Tuple[str, float]]]:
        if self.multi_engines:
            logging.info("Multi-engine mode is enabled. Using jieba to extract keywords.")
            processed_text = ' '.join(self.cut(text))
            if self.keywords_method == 'tfidf':
                return analyse.extract_tags(processed_text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
            elif self.keywords_method == 'textrank':
                return analyse.textrank(processed_text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
        raise HanSegError("Multi-engine mode is disabled and thulac does not support keywords extract. You can set multi_engines=true in config.")

    def sentiment_analysis(self, text: str) -> float:
        if self.multi_engines:
            logging.info("Multi-engine mode is enabled. Using snownlp to perform sentiment analysis.")
            return SnowNLP(text).sentiments
        raise HanSegError(f"Multi-engine mode is disabled and {self.engine_name} does not support this method. You can set multi_engines=true in config.")
        
    def _reload_engine(self) -> None:
        self._thulac = thulac(
            model_path=self.model_path,
            seg_only=self.seg_only,
            T2S=self.t2s,
            user_dict=self.user_dict_path
        )