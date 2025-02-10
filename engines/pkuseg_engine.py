from typing import List, Tuple, Union
from base import HanSegBase, HanSegError, logger
from jieba import analyse


class HanSegPkuseg(HanSegBase):
    """Implementation based on pkuseg."""
    def __init__(self, global_config: dict = {}, local_config: dict = {}):
        from pkuseg import pkuseg
        super().__init__(global_config, local_config)
        
        self.model_name = local_config.get('model_name', 'default')
        self.postag = local_config.get('postag', True)
        self._pkuseg = pkuseg(model_name=self.model_name, user_dict=self.user_dict_path, postag=self.postag)
        
    def cut(self, text: str) -> List[str]:
        if self.postag:
            if self.filt:
                return [word[0] for word in self._pkuseg.cut(text) if word[0] not in self.stop_words]
            return [word[0] for word in self._pkuseg.cut(text)] 
        
        if self.filt:
            return [word for word in self._pkuseg.cut(text) if word not in self.stop_words]
        return self._pkuseg.cut(text)
    
    def pos(self, text: str) -> List[Tuple[str, str]]:
        if not self.postag:
            raise HanSegError("postag is not enabled in config.")
        if self.filt:
            return [(word[0], word[1]) for word in self._pkuseg.cut(text) if word[0] not in self.stop_words]
        return self._pkuseg.cut(text)

    def keywords(self, text: str) -> Union[List[str], List[Tuple[str, float]]]:
        if self.multi_engines:
            logger.info("Multi-engine mode is enabled. Using jieba to extract keywords.")
            processed_text = ' '.join(self.cut(text))
            if self.keywords_method == 'tfidf':
                return analyse.extract_tags(processed_text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
            elif self.keywords_method == 'textrank':
                return analyse.textrank(processed_text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
        else:
            raise HanSegError("Multi-engine mode is disabled and pkuseg does not support this method. You can set multi_engines=true in config.")