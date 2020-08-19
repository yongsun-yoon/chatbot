from typing import Text, List, Optional, Dict, Any, Tuple

import regex
import re

from rasa.nlu.config import RasaNLUModelConfig
from rasa.constants import DOCS_URL_COMPONENTS
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.training_data import TrainingData, Message
import rasa.utils.common as common_utils

from rasa.nlu.constants import (
    RESPONSE,
    TEXT,
    CLS_TOKEN,
    TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
    INTENT,
)

## custom constant
MECAB_DIR = 'mecab_dir'
POS_NAME = 'pos'

from konlpy.tag import Mecab

class MecabTokenizer(Tokenizer):

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Regular expression to detect tokens
        "token_pattern": None,
    }

    # the following language should not be tokenized using the WhitespaceTokenizer
    not_supported_language_list = []

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework."""

        super().__init__(component_config)

        self.mecab = Mecab(self.component_config[MECAB_DIR])

        if "case_sensitive" in self.component_config:
            common_utils.raise_warning(
                "The option 'case_sensitive' was moved from the tokenizers to the "
                "featurizers.",
                docs=DOCS_URL_COMPONENTS,
            )

    def tokenize(self, message: Message, attribute: Text) -> Tuple[List[Token], List[str]]:
        text = message.get(attribute)

        # we need to use regex instead of re, because of
        # https://stackoverflow.com/questions/12746458/python-unicode-regular-expression-matching-failing-with-some-unicode-characters

        # remove 'not a word character' if
        text = re.sub('[^0-9ㄱ-힣]', ' ', text) # remove special words
        results = self.mecab.pos(text)
        if len(results) > 0:
            words, pos = zip(*results)
        else:
            words, pos = None, None

        # if we removed everything like smiles `:)`, use the whole text as 1 token
        if not words:
            words = [text]
            pos = ['NNG']

        tokens = self._convert_words_to_tokens(words, text)
        tokens = list(tokens)
        pos = list(pos)
        return tokens, pos

    def train(self, training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any) -> None:
        """Tokenize all training data."""

        for example in training_data.training_examples:
            for attribute in MESSAGE_ATTRIBUTES:
                if example.get(attribute) is not None:
                    if attribute == INTENT:
                        tokens = self._split_intent(example)
                    else:
                        tokens, pos = self.tokenize(example, attribute)
                        tokens, pos = self.add_cls_token(tokens, pos, attribute)
                    example.set(TOKENS_NAMES[attribute], tokens)
                    example.set(POS_NAME, pos)

    def process(self, message: Message, **kwargs: Any) -> None:
        """Tokenize the incoming message."""

        tokens, pos = self.tokenize(message, TEXT)
        tokens, pos = self.add_cls_token(tokens, pos, TEXT)
        message.set(TOKENS_NAMES[TEXT], tokens)
        message.set(POS_NAME, pos)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        return

    @staticmethod
    def add_cls_token(tokens: List[Token], pos:List[str], attribute: Text) -> Tuple[List[Token], List[str]]:
        if attribute in [RESPONSE, TEXT] and tokens:
            # +1 to have a space between the last token and the __cls__ token
            idx = tokens[-1].end + 1
            tokens.append(Token(CLS_TOKEN, idx))
            pos.append('SPECIAL')
        return tokens, pos

    
