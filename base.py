# base.py

from typing import List, TextIO, Tuple, Set, Union
from jieba import analyse
import os
import yaml
import logging

def check_and_get_stop_words(config: dict) -> Set[str]:
    """
    Check if the stop words file exists, and return the set of stop words.
    """
    stop_words_path = config.get('stop_words', '')
    if not stop_words_path:
        raise HanSegError("Stop words file path is not specified in the config file when filt=true.")
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stop_words = {line.strip() for line in f if line.strip()}
    return stop_words_path, stop_words

def load_config(config_path: str) -> dict:
    """
    Load config from yaml file.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class HanSegBase:
    def __init__(self, global_config: dict = {}, local_config: dict = {}):

        self.global_config = global_config or {}
        self.local_config = local_config or {}
        self.engine_name = local_config.get('engine_name', None)
        if not self.engine_name:
            raise HanSegError("Engine name is not specified in the config.")

        self.multi_engines = global_config.get('multi_engines', True)
        self.filt = self.local_config.get('filt', False)

        if self.engine_name != 'snownlp':
            self.user_dict_path = self.local_config.get('user_dict', '')
            if self.engine_name == 'pkuseg':
                if not self.user_dict_path:
                    self.user_dict_path = 'default'
            elif not os.path.exists(self.user_dict_path):
                raise HanSegError(f"User dictionary file {self.user_dict_path} not found.\nIf you don't need to set a user dict, leave user_dict an empty string in your config.")

        self.stop_words = set()
        if self.filt:
            stop_words_path, self.stop_words = check_and_get_stop_words(local_config)
            if self.multi_engines or self.engine_name == 'jieba':
                analyse.set_stop_words(stop_words_path)

        self.topK = self.local_config.get('topK', 20)

        if self.engine_name != 'snownlp':
            self.keywords_method = self.local_config.get('keywords_method', '').lower()
            if self.keywords_method not in ('tfidf', 'textrank') and self.multi_engines:
                raise HanSegError(f"You must set keywords_method to 'tfidf' or 'textrank' in your config.")

            self.withWeight = self.local_config.get('withWeight', False)
            self.allowPOS_config = self.local_config.get('allowPOS', None)
            self.allowPOS = tuple(self.allowPOS_config.split()) if self.allowPOS_config else ()

            self.idf_path = self.local_config.get('idf_path', None)
            if self.keywords_method == 'tfidf' and self.idf_path and (self.multi_engines or self.engine_name == 'jieba'):
                analyse.set_idf_path(self.idf_path)
     
        self.batch_size = self.global_config.get('cut_file_batch_size', 100)

    def cut(self, text: str) -> List[str]:
        raise HanSegError(f"Engine '{self.engine_name}' does not support this method.")

    def pos(self, text: str) -> List[Tuple[str, str]]:
        raise HanSegError(f"Engine '{self.engine_name}' does not support this method.")

    def add_word(self, word: str, freq: int = 1, flag: str = None) -> None:
        if self.engine_name != 'snownlp':
            word = word.strip()
            flag = flag.strip() if flag else None
            if word:
                line = f"{word} {flag}" if flag else word
                with open(self.user_dict_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{line}\n")
                    self._reload_engine()
        else:
            raise HanSegError(f"Engine '{self.engine_name}' does not support this method.")

    def del_word(self, word: str) -> None:
        if self.engine_name != 'snownlp':
            with open(self.user_dict_path, 'r', encoding='utf-8') as f:
                lines = []
                for line in f:
                    lst = line.split()
                    if lst and lst[0] != word:
                        lines.append(line)
                        
            with open(self.user_dict_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
            self._reload_engine()
        else:
            raise HanSegError(f"Engine '{self.engine_name}' does not support this method.")

    def suggest_freq(self, words) -> None:
        raise HanSegError(f"Engine '{self.engine_name}' does not support this method.")

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
        raise HanSegError(f"Engine '{self.engine_name}' does not support this method.")
    
    def cut_file(self, input_path: str, output_path: str) -> None:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:
            batch = []
            for line in f_in:
                line = line.strip()
                if line:
                    batch.append(line)
                    if len(batch) >= self.batch_size:
                        processed = [' '.join(self.cut(l)) + '\n' for l in batch]
                        f_out.writelines(processed)
                        batch = []
            if batch:
                processed = [' '.join(self.cut(l)) + '\n' for l in batch]
                f_out.writelines(processed)

    def _reload_engine(self) -> None:
        raise NotImplementedError
    
class HanSegError(Exception):
    pass