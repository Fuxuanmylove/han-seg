# han_seg/interface.py

import os
from typing import List, Tuple, Set
import yaml
import jieba
from jieba import analyse
from jieba import posseg as pseg
import logging
from snownlp import SnowNLP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def check_and_get_stop_words(config: dict) -> Set[str]:
    """
    Check if the stop words file exists, and return the set of stop words.
    """
    stop_words_path = config.get('stop_words_path', '')
    if not stop_words_path:
        raise HanSegError("Stop words file path is not specified in the config file when filt=true.")
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stop_words = {line.strip() for line in f if line.strip()}
    return stop_words_path, stop_words

class HanSeg:

    def __init__(self, engine: str = 'jieba', config_path: str = "config.yaml"):
        """
        :param engine: 'jieba'/'thulac'
        :param config_path: config file path
        """
        config = self._load_config(config_path)
        self.global_config = config.get('global', {})
        self.multi_engines = self.global_config.get('multi_engines', True)
        self.engine_name = engine.lower()
        if self.engine_name == 'jieba':
            local_config = config.get('jieba', {})
            self._engine = HanSegJieba(self.global_config, local_config)
        elif self.engine_name == 'thulac':
            local_config = config.get('thulac', {})
            self._engine = HanSegThulac(self.global_config, local_config)
        else:
            raise HanSegError(f"Engine '{engine}' is not supported. Supported engines: jieba, thulac.")
            
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

    def keyword_extract(self, text: str):
        """
        Keywords extract method, return a list of keywords or (keyword, weight) tuples, depends on the config.
        """
        return self._engine.keyword_extract(text)
    
    def _load_config(self, config_path: str):
        """
        Load config from yaml file.
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
class HanSegJieba:
    """Implementation based on jieba."""
    def __init__(self, global_config: dict = None, local_config: dict = None):
        self.global_config = global_config or {}
        self.local_config = local_config or {}
        self.HMM = local_config.get('HMM', True)
        self.filt = local_config.get('filt', False)
        self.tune = local_config.get('tune', True)
        self.topK = local_config.get('topK', 20)
        self.withWeight = local_config.get('withWeight', False)

        self.allowPOS_config = local_config.get('allowPOS', None)
        self.allowPOS = tuple(self.allowPOS_config.split()) if self.allowPOS_config else ()

        self.dictionary_path = local_config.get('dictionary', None)
        if self.dictionary_path:
            if not os.path.exists(self.dictionary_path):
                raise HanSegError(f"Dictionary file {self.dictionary_path} not found.\nIf you don't need to set a dictionary, leave dictionary an empty string in your config.")
            jieba.set_dictionary(self.dictionary_path)
 
        self.user_dict_path = local_config.get('user_dict', None)
        if self.user_dict_path:
            if not os.path.exists(self.user_dict_path):
                raise HanSegError(f"User dictionary file {self.user_dict_path} not found.\nIf you don't need to set a user dictionary, leave user_dict an empty string in your config.")
            jieba.load_userdict(self.user_dict_path)

        self.stop_words = set()
        if self.filt:
            stop_words_path, self.stop_words = check_and_get_stop_words(local_config)
            analyse.set_stop_words(stop_words_path)

        self.keyword_extract_method = self.local_config.get('keyword_extract_method', '').lower()
        if self.keyword_extract_method not in ('tfidf', 'textrank'):
            raise HanSegError(f"You must set keyword_extract_method to 'tfidf' or 'textrank' in your config.")

        self.cut_mode = local_config.get('cut_mode', 'default').lower()
        if self.cut_mode not in ('default', 'full', 'search'):
            raise HanSegError(f"Invalid cut mode: {self.cut_mode}.\nYou must set cut_mode to 'default', 'full' or 'search' in your config.")
        
        self.idf_path = self.local_config.get('idf_path', None)
        if self.keyword_extract_method == 'tfidf' and self.idf_path:
            if not os.path.exists(self.idf_path):
                raise HanSegError(f"IDF file {self.idf_path} not found.\nIf you don't need to set IDF, leave idf_path an empty string in your config.")
            analyse.set_idf_path(self.idf_path)

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
        return [tuple(i) for i in pseg.lcut(text, HMM=self.HMM)]
        
    def add_word(self, word: str, freq: int = 1, flag: str = None):
        jieba.add_word(word, freq, flag)
        
    def del_word(self, word: str):
        jieba.del_word(word)
        
    def suggest_freq(self, words):
        jieba.suggest_freq(words, tune=self.tune)
        
    def keyword_extract(self, text):
        if self.keyword_extract_method == 'tfidf':
            return analyse.extract_tags(text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
        elif self.keyword_extract_method == 'textrank':
            return analyse.textrank(text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)

class HanSegThulac:
    """Implementation based on thulac."""
    def __init__(self, global_config: dict = None, local_config: dict = None):
        from thulac import thulac
        self.multi_engines = global_config.get('multi_engines', True)
        self.global_config = global_config or {}
        self.local_config = local_config or {}
        model_path = self.local_config.get('model_path', None)
        if not model_path:
            model_path = None
        self.seg_only = self.local_config.get('seg_only', False)
        self.t2s = self.local_config.get('t2s', False)
        self.filt = self.local_config.get('filt', False)
        self.user_dict = self.local_config.get('user_dict', None)
        if self.user_dict and not os.path.exists(self.user_dict):
            raise HanSegError(f"User dictionary file {self.user_dict} not found.\nIf you don't need to set a user dict, leave user_dict an empty string in your config.")
        self._thulac = thulac(model_path=model_path, seg_only=self.seg_only, T2S=self.t2s, user_dict=self.user_dict)
        self.stop_words = set()
        if self.filt:
            stop_words_path, self.stop_words = check_and_get_stop_words(local_config)
            if self.multi_engines:
                analyse.set_stop_words(stop_words_path)
        
        self.keyword_extract_method = self.local_config.get('keyword_extract_method', '').lower()
        if self.keyword_extract_method not in ('tfidf', 'textrank'):
            raise HanSegError(f"You must set keyword_extract_method to 'tfidf' or 'textrank' in your config.")
        self.topK = self.local_config.get('topK', 20)
        self.withWeight = self.local_config.get('withWeight', False)
        
        self.allowPOS_config = self.local_config.get('allowPOS', None)
        self.allowPOS = tuple(self.allowPOS_config.split()) if self.allowPOS_config else ()

        self.idf_path = self.local_config.get('idf_path', None)
        if self.keyword_extract_method == 'tfidf' and self.idf_path and self.multi_engines:
            if not os.path.exists(self.idf_path):
                raise HanSegError(f"IDF file {self.idf_path} not found.\nIf you don't need to set IDF, leave idf_path an empty string in your config.")
            analyse.set_idf_path(self.idf_path)
            
    def cut(self, text: str) -> List[str]:
        if self.filt:
            return [word[0] for word in self._thulac.cut(text) if word[0] not in self.stop_words]
        else:
            return [word[0] for word in self._thulac.cut(text)]
    
    def pos(self, text: str) -> List[Tuple[str, str]]:
        if self.seg_only:
            raise HanSegError("You can't get POS when seg_only is True. Set seg_only=false in config.")
        pos_result = []
        if self.filt:
            for word, pos in self._thulac.cut(text):
                if word not in self.stop_words:
                    pos_result.append((word, pos))
        else:
            pos_result = self._thulac.cut(text)
            pos_result = [(word, pos) for word, pos in pos_result]
        return pos_result

    def add_word(self, word: str = None, freq: int = None, tag: str = None) -> None:
        """thulac does not support adding words."""
        raise HanSegError("thulac engine does not support dynamically adding words.")
        
    def del_word(self, word: str = None) -> None:
        """thulac does not support deleting words."""
        raise HanSegError("thulac engine does not support dynamically deleting words.")

    def keyword_extract(self, text: str) -> List[str]:
        if self.multi_engines:
            logging.info("Multi-engine mode is enabled. Using jieba to extract keywords.")
            words = self.cut(text)
            processed_text = ' '.join(words)
            if self.keyword_extract_method == 'tfidf':
                return analyse.extract_tags(processed_text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
            elif self.keyword_extract_method == 'textrank':
                return analyse.textrank(processed_text, topK=self.topK, withWeight=self.withWeight, allowPOS=self.allowPOS)
        else:
            raise HanSegError("Multi-engine mode is disabled and thulac does not support keyword extract. You can enable it by setting multi_engines=true in config.")

class HanSegPKUSeg:
    """Implementation based on pkuseg."""
    def __init__(self, global_config: dict = None, local_config: dict = None):
        from pkuseg import pkuseg
        self.global_config = global_config or {}
        self.local_config = local_config or {}
        
        self.multi_engines = global_config.get('multi_engines', True)
        self.model_name = local_config.get('model_name', 'default')
        self.user_dict = local_config.get('user_dict', 'default')
        self.postag = local_config.get('postag', True)
        self.filt = local_config.get('filt', False)
        
        self.seg = pkuseg(model_name=self.model_name, user_dict=self.user_dict, postag=self.postag)

        self.stop_words = set()
        if self.filt:
            stop_words_path, self.stop_words = check_and_get_stop_words(local_config)
            if self.multi_engines:
                analyse.set_stop_words(stop_words_path)
            
        
class HanSegError(Exception):
    pass
