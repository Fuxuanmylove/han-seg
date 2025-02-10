from typing import List, Tuple
from base import HanSegBase, HanSegError, logger
from snownlp import SnowNLP

class HanSegSnowNLP(HanSegBase):
    """Implementation based on SnowNLP"""
    def __init__(self, global_config: dict = {}, local_config: dict = {}):
        super().__init__(global_config, local_config)
    
    def cut(self, text: str) -> List[str]:
        if self.filt:
            return [word for word in SnowNLP(text).words if word not in self.stop_words]
        return SnowNLP(text).words
    
    def keywords(self, text: str) -> List[str]:
        if self.filt:
            return [word for word in SnowNLP(text).keywords(self.topK) if word not in self.stop_words]
        return SnowNLP(text).keywords(self.topK)
    
    def sentiment_analysis(self, text: str) -> float:
        return SnowNLP(text).sentiments