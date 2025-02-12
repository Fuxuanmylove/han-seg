# base.py

from collections import Counter
from typing import List, Tuple, Set, Union
from jieba import analyse
import os
import yaml
import logging
from snownlp import SnowNLP
import hanlp
from hanlp.pretrained.sts import STS_ELECTRA_BASE_ZH


class HanSegBase:
    def __init__(self, engine_name: str, multi_engines: bool, user_dict: str, filt: bool, stop_words_path: str, local_config: dict):

        self.local_config = local_config or {}
        self.engine_name = engine_name
        self.multi_engines = multi_engines
        self.filt = filt
        self.user_dict_path = user_dict

        self._initialize_user_dict()

        self.stop_words = set()
        if self.filt:
            self.stop_words_path = stop_words_path
            if self.stop_words_path is not None:
                if not os.path.exists(self.stop_words_path):
                    raise HanSegError(f"Stop words file {self.stop_words_path} not found.\nIf you don't need it, please set filt to False.")
                self._clean_file(self.stop_words_path)
                if self.multi_engines or self.engine_name == 'jieba':
                    analyse.set_stop_words(self.stop_words_path)
                self.stop_words = HanSegBase._check_and_get_stop_words(self.stop_words_path)

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

    def cut(self, texts: List[str], with_position: bool = False) -> Union[List[List[str]], List[List[Tuple[str, int, int]]]]:
        raise HanSegError(f"Engine '{self.engine_name}' does not support this method.")

    def pos(self, text: str) -> List[Tuple[str, str]]:
        raise HanSegError(f"Engine '{self.engine_name}' does not support this method.")

    def add_word(self, word: str, freq: int = 1, tag: str = None) -> None:
        if not self.user_dict_path:
            raise HanSegError("User dict is not set.")
        word = word.strip()
        tag = tag.strip() if tag else None
        if word:
            line = f"{word} {tag}" if tag else word
            with open(self.user_dict_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{line}\n")

    def del_word(self, word: str) -> None:
        if not self.user_dict_path:
            raise HanSegError("User dict is not set.")
        with open(self.user_dict_path, 'r', encoding='utf-8') as f:
            lines = []
            for line in f:
                lst = line.split()
                if lst and lst[0] != word:
                    lines.append(line)       
        with open(self.user_dict_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    def suggest_freq(self, words) -> None:
        raise HanSegError(f"Engine '{self.engine_name}' does not support this method.")

    def keywords(self, text: str, limit: int = 10) -> Union[List[str], List[Tuple[str, float]]]:
        if self.multi_engines:
            logging.info("Multi-engine mode is enabled. Using jieba to extract keywords.")
            processed_text = ' '.join(self.cut([text])[0])
            print(processed_text)
            if self.keywords_method == 'tfidf':
                return analyse.extract_tags(processed_text, topK=limit, withWeight=self.withWeight, allowPOS=self.allowPOS)
            elif self.keywords_method == 'textrank':
                return analyse.textrank(processed_text, topK=limit, withWeight=self.withWeight, allowPOS=self.allowPOS)
        raise HanSegError(f"Multi-engine mode is disabled and {self.engine_name} does not support keywords extract.")

    def sentiment_analysis(self, text: str) -> float:
        if self.multi_engines:
            logging.info("Multi-engine mode is enabled. Using snownlp to perform sentiment analysis.")
            processed_text = ' '.join(self.cut([text])[0])
            score = SnowNLP(processed_text).sentiments
            if score >= 0.5:
                return f"{score:.5f} (Positive)"
            return f"{(1 - score):.5f} (Negative)"
        raise HanSegError(f"Multi-engine mode is disabled and {self.engine_name} does not support this method.")

    def cut_file(self, input_path: str, output_path: str, batch_size: int = 1000) -> None:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:
            batch = []
            for line in f_in:
                stripped_line = line.strip()
                if stripped_line:
                    batch.append(stripped_line)
                    if len(batch) == batch_size:
                        raw_result = self.cut(batch)
                        lines = "\n".join(" ".join(words) for words in raw_result)
                        f_out.writelines(lines)
                        batch = []
            if batch:
                raw_result = self.cut(batch)
                lines = "\n".join(" ".join(words) for words in raw_result)
                f_out.writelines(lines)

    def words_count(self, input_file: str, output_file: str) -> None:
        word_counts = Counter()
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    words = self.cut([line])[0]
                    word_counts.update(words)
        with open(output_file, 'w', encoding='utf-8') as f:
            for word, count in word_counts.most_common():
                f.write(f"{word} {count}\n")

    def reload_engine(self) -> None:
        self._initialize_user_dict()

    def _clean_file(self, file_path: str) -> None:
        with open(file_path, 'r+', encoding='utf-8') as f:
            seen = set()
            kept_lines = []
            for line in f:
                word = line.strip()
                if word and word not in seen:
                    seen.add(word)
                    kept_lines.append(word + '\n')
            f.seek(0)
            f.writelines(kept_lines)
            f.truncate()

    def _deal_with_raw_cut_result(self, result: List[List[str]], with_position: bool = False) -> Union[List[List[str]], List[List[Tuple[str, int, int]]]]:    
        if with_position:
            result = [HanSegBase._add_position(words) for words in result]
        if self.filt:
            if with_position:            
                result = [[(word, start, end) for word, start, end in words if word not in self.stop_words] for words in result]
            else:
                result = [[word for word in words if word not in self.stop_words] for words in result]
        return result

    def _initialize_user_dict(self) -> None:
        if not (self.engine_name == 'pkuseg' and self.user_dict_path == 'default'):
            if self.user_dict_path is not None:
                if not os.path.exists(self.user_dict_path):
                    raise HanSegError(f"User dictionary file {self.user_dict_path} not found.")
                self._clean_file(self.user_dict_path)

    @staticmethod
    def _check_and_get_stop_words(stop_words_path: str) -> Set[str]:
        """Check if the stop words file exists, and return the set of stop words."""
        with open(stop_words_path, 'r', encoding='utf-8') as f:
            stop_words = {line.strip() for line in f if line.strip()}
        return stop_words

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load config from yaml file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    def _add_position(words: List[str]) -> List[Tuple[str, int, int]]:
        """Add position information to each word. Please use it before filtering stop words."""
        result = []
        start = 0
        for word in words:
            end = start + len(word)
            result.append((word, start, end))
            start = end
        return result

class HanSegError(Exception):
    pass