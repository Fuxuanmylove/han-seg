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
        
        super().add_word(word, freq, flag)
        
    def del_word(self, word: str) -> None:
        if not self.user_dict_path:
            raise HanSegError("user_dict is not set in config.")

        super().del_word(word)

    def keywords(self, text: str, limit: int = 10) -> Union[List[str], List[Tuple[str, float]]]:
        return super().keywords(text, limit)

    def sentiment_analysis(self, text: str) -> float:
        return super().sentiment_analysis(text)
  
    def _reload_engine(self) -> None:
        self._thulac = thulac(
            model_path=self.model_path,
            seg_only=self.seg_only,
            T2S=self.t2s,
            user_dict=self.user_dict_path
        )