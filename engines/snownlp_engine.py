from typing import List, Tuple
from base import HanSegBase, HanSegError
from snownlp import SnowNLP

class HanSegSnowNLP(HanSegBase):
    """Implementation based on SnowNLP"""
    def __init__(self, engine_name: str, filt: bool, global_config: dict, local_config: dict):
        super().__init__(engine_name, filt, global_config, local_config)
    
    def cut(self, text: str) -> List[str]:
        if self.filt:
            return [word for word in SnowNLP(text).words if word not in self.stop_words]
        return SnowNLP(text).words
    
    def pos(self, text: str) -> List[Tuple[str, str]]:
        if self.filt:
            return [(word, tag) for word, tag in SnowNLP(text).tags if word not in self.stop_words]
        return [(word, tag) for word, tag in SnowNLP(text).tags]
    
    def keywords(self, text: str, limit: int = 10) -> List[str]:
        if self.filt:
            return [word for word in SnowNLP(text).keywords(limit) if word not in self.stop_words]
        return SnowNLP(text).keywords(limit)
    
    def sentiment_analysis(self, text: str) -> float:
        return SnowNLP(text).sentiments