import logging
from typing import Any, Dict, List, Optional, Text, Tuple, Type
from tqdm import tqdm

from rasa.constants import DOCS_URL_COMPONENTS
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    TEXT,
    TOKENS_NAMES,
    DENSE_FEATURE_NAMES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
)
import numpy as np
import tensorflow as tf

import rasa.utils.train_utils as train_utils
import rasa.utils.common as common_utils

import os
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
from .tokenizer import MecabTokenizer

logger = logging.getLogger(__name__)

## custom constant
MODEL = 'model'
MODEL_PATH = 'model_path'
POS_NAME = 'pos'
TRAIN = 'train'
MODEL_SIZE = 'model_size'
WINDOW_SIZE = 'window_size'
MIN_COUNT = 'min_count'
EPOCHS = 'epochs'

class WordEmbedFeaturizer(DenseFeaturizer):
    """Featurizer using Word2Vec/FastText model.
    Loads the model and computes sentence and sequence level feature representations 
    for dense featurizable attributes of each message object.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [MecabTokenizer]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super(WordEmbedFeaturizer, self).__init__(component_config)
        if self.component_config[MODEL] == 'fasttext':
            self.model_class = FastText
        elif self.component_config[MODEL] == 'word2vec':
            self.model_class = Word2Vec

        if not self.component_config[TRAIN]:
            self.load_model()


    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["gensim"]

    def load_model(self) -> None:
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        model_path = os.path.join(base_dir, self.component_config[MODEL_PATH])
        self.model = FastText.load(model_path)

    def get_data_from_examples(self, examples: List[Message], attribute: Text = TEXT) -> List[List[str]]:
        list_of_tokens = [example.get(TOKENS_NAMES[attribute]) for example in examples]
        list_of_pos = [example.get(POS_NAME) for example in examples]        
        data = []
        for i, j in zip(list_of_tokens, list_of_pos):
            data.append([_i.text + '_' + _j for _i, _j in zip(i, j)])
        return data

    def _combine_encodings(
        self,
        sentence_encodings: np.ndarray,
        sequence_encodings: np.ndarray,
        number_of_tokens_in_sentence: List[int],
    ) -> np.ndarray:
        """Combine the sequence encodings with the sentence encodings.

        Append the sentence encoding to the end of the sequence encodings (position
        of CLS token)."""

        final_embeddings = []

        for index in range(len(number_of_tokens_in_sentence)):
            sequence_length = number_of_tokens_in_sentence[index]
            sequence_encoding = sequence_encodings[index][:sequence_length]
            sentence_encoding = sentence_encodings[index]
            sequence_encoding[-1] = sentence_encoding
            final_embeddings.append(sequence_encoding)
        
        return np.array(final_embeddings)

    def _compute_features(self, batch_examples: List[Message], attribute: Text = TEXT) -> Tuple[np.ndarray, List[int]]:

        batch_data = self.get_data_from_examples(batch_examples, attribute)
        sequence_encodings = [[self.model.wv[token] if token in self.model.wv.vocab else np.random.rand(self.model.wv.vector_size) for token in sentence] for sentence in batch_data]
        sequence_encodings = [np.array(i) for i in sequence_encodings]
        sentence_encodings = [np.mean(i, axis=0, keepdims=True) for i in sequence_encodings] # mean pooling
        # sentence_encodings = [np.max(i, axis=0, keepdims=True) for i in sequence_encodings] # max pooling

        number_of_tokens_in_sentence = [len(sentence) for sentence in batch_data]
        return self._combine_encodings(sentence_encodings, sequence_encodings, number_of_tokens_in_sentence)


    def _train(self, training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any) -> None:
        non_empty_examples = []
        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            non_empty_examples += list(filter(lambda x: x.get(attribute), training_data.training_examples))
        
        data = self.get_data_from_examples(non_empty_examples)
        model = self.model_class(
                    data,
                    size=self.component_config[MODEL_SIZE], 
                    window=self.component_config[WINDOW_SIZE], 
                    min_count=self.component_config[MIN_COUNT],
                    iter=self.component_config[EPOCHS])
                    
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        model_path = os.path.join(base_dir, self.component_config[MODEL_PATH])
        model.save(model_path)
    
    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        if self.component_config[TRAIN]:
            print('Dense Featurizer Training...')
            self._train(training_data, config, **kwargs)
            print('Dense Featurizer Training Completed')

        batch_size = 64
        self.load_model()

        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            non_empty_examples = list(filter(lambda x: x.get(attribute), training_data.training_examples))
            progress_bar = tqdm(range(0, len(non_empty_examples), batch_size), desc=attribute.capitalize() + " batches")

            for batch_start_index in progress_bar:
                batch_end_index = min(batch_start_index + batch_size, len(non_empty_examples))

                # Collect batch examples
                batch_examples = non_empty_examples[batch_start_index:batch_end_index]
                batch_features = self._compute_features(batch_examples, attribute)

                for index, ex in enumerate(batch_examples):
                    ex.set(
                        DENSE_FEATURE_NAMES[attribute],
                        self._combine_with_existing_dense_features(
                            ex, batch_features[index], DENSE_FEATURE_NAMES[attribute]
                        ),
                    )

    def process(self, message: Message, **kwargs: Any) -> None:
        features = self._compute_features([message])[0]
        message.set(
            DENSE_FEATURE_NAMES[TEXT],
            self._combine_with_existing_dense_features(
                message, features, DENSE_FEATURE_NAMES[TEXT]
            ),
        )
