# hanseg/interface.py

from .base import HanSegBase, HanSegError, check_and_get_stop_words, load_config, get_logger
from typing import List, Tuple
import jieba
from jieba import analyse
from jieba import posseg as pseg
from snownlp import SnowNLP

config = load_config("config.yaml")

log_config = config.get('log', {'name': 'hanseg', 'level': 'info', 'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'})
logger = get_logger(log_config)

class HanSeg:

    def __init__(self, engine_name: str = 'jieba'):
        """
        :param engine: jieba / thulac / pkuseg
        """
        self.global_config = config.get('global', {})
        self.engine_name = engine_name.lower()
        if self.engine_name == 'jieba':
            local_config = config.get('jieba', {})
            self._engine = HanSegJieba(self.global_config, local_config)
        elif self.engine_name == 'thulac':
            local_config = config.get('thulac', {})
            self._engine = HanSegThulac(self.global_config, local_config)
        elif self.engine_name == 'pkuseg':
            local_config = config.get('pkuseg', {})
            self._engine = HanSegPkuseg(self.global_config, local_config)
        else:
            raise HanSegError(f"Engine '{engine_name}' is not supported. Supported engines: jieba, thulac, pkuseg.")
            
    def cut(self, text: str) -> List[str]:
        """
        Standard cut method, return a list of words.
        """
        return self._engine.cut(text)
    
    def pos(self, text: str) -> List[Tuple[str, str]]:
        """
        Returns the tokens and their corresponding POS tags.
        """
        return self._engine.pos(text)
        
    def add_word(self, word: str, freq: int = 1, tag: str = None):
        """
        Dynamically add words, if supported by the engine.
        """
        self._engine.add_word(word, freq, tag)

    def del_word(self, word: str):
        """
        Dynamically delete words, if supported by the engine.
        """
        self._engine.del_word(word)

    def keywords(self, text: str):
        """
        Keywordss extract method, return a list of keywordss or (keywords, weight) tuples, depends on the config.
        """
        return self._engine.keywords(text)
    
class HanSegJieba(HanSegBase):
    """Implementation based on jieba."""
    def __init__(self, global_config: dict = None, local_config: dict = None):
        
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
        
    def keywords(self, text: str) -> List[str]:
        if self.keywords_method == 'tfidf':
            return analyse.extract_tags(text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
        elif self.keywords_method == 'textrank':
            return analyse.textrank(text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)

class HanSegThulac(HanSegBase):
    """Implementation based on thulac."""
    def __init__(self, global_config: dict = None, local_config: dict = None):
        from thulac import thulac
        super().__init__(global_config, local_config)

        model_path = self.local_config.get('model_path', None)
        if not model_path:
            model_path = None
        self.seg_only = self.local_config.get('seg_only', False)
        self.t2s = self.local_config.get('t2s', False)
        self._thulac = thulac(model_path=model_path, seg_only=self.seg_only, T2S=self.t2s, user_dict=self.user_dict_path)
            
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

    def keywords(self, text: str) -> List[str]:
        if self.multi_engines:
            logger.info("Multi-engine mode is enabled. Using jieba to extract keywords.")
            processed_text = ' '.join(self.cut(text))
            if self.keywords_method == 'tfidf':
                return analyse.extract_tags(processed_text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
            elif self.keywords_method == 'textrank':
                return analyse.textrank(processed_text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
        else:
            raise HanSegError("Multi-engine mode is disabled and thulac does not support keywords extract. You can set multi_engines=true in config.")

class HanSegPkuseg(HanSegBase):
    """Implementation based on pkuseg."""
    def __init__(self, global_config: dict = None, local_config: dict = None):
        from pkuseg import pkuseg
        super().__init__(global_config, local_config)
        
        self.model_name = local_config.get('model_name', 'default')
        self.postag = local_config.get('postag', True)
        self._pkuseg = pkuseg(model_name=self.model_name, user_dict=self.user_dict_path, postag=self.postag)

        self.stop_words = set()
        if self.filt:
            stop_words_path, self.stop_words = check_and_get_stop_words(local_config)
            if self.multi_engines:
                analyse.set_stop_words(stop_words_path)
        
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

    def keywords(self, text: str) -> List[str]:
        if self.multi_engines:
            logger.info("Multi-engine mode is enabled. Using jieba to extract keywords.")
            processed_text = ' '.join(self.cut(text))
            if self.keywords_method == 'tfidf':
                return analyse.extract_tags(processed_text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
            elif self.keywords_method == 'textrank':
                return analyse.textrank(processed_text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
        else:
            raise HanSegError("Multi-engine mode is disabled and pkuseg does not support this method. You can set multi_engines=true in config.")