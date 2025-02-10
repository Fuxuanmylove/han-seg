import logging
from typing import List, Tuple, Union
from base import HanSegBase, HanSegError
from jieba import analyse
from pkuseg import pkuseg
from snownlp import SnowNLP


class HanSegPkuseg(HanSegBase):
    """Implementation based on pkuseg."""
    def __init__(self, global_config: dict = {}, local_config: dict = {}):
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
    
    def add_word(self, word: str, freq: int = 1, flag: str = None) -> None:
        #TODO: immediate / after close
        if not self.user_dict_path:
            raise HanSegError("user_dict is not set in config.")
        
        if self.user_dict_path == 'default':
            raise HanSegError("You cannot modify the default user_dict.")
        
        word = word.strip()
        flag = flag.strip() if flag else None
        with open(self.user_dict_path, 'a', encoding='utf-8') as f:
            line = f"{word} {flag}" if flag else word
            f.write(line + '\n')
            
        self._reload_engine()

    def del_word(self, word: str) -> None:
        if not self.user_dict_path:
            raise HanSegError("user_dict is not set in config.")
        
        if self.user_dict_path == 'default':
            raise HanSegError("You cannot modify the default user_dict.")
        
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
        raise HanSegError("Multi-engine mode is disabled and pkuseg does not support this method. You can set multi_engines=true in config.")
        
    def sentiment_analysis(self, text: str) -> float:
        if self.multi_engines:
            logging.info("Multi-engine mode is enabled. Using snownlp to perform sentiment analysis.")
            return SnowNLP(text).sentiments
        raise HanSegError(f"Multi-engine mode is disabled and {self.engine_name} does not support this method. You can set multi_engines=true in config.")
    
    def _reload_engine(self) -> None:
        self._pkuseg = pkuseg(
            model_name=self.model_name,
            user_dict=self.user_dict_path,
            postag=self.postag
        )