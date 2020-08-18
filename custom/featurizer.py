import logging
from typing import Any, Dict, List, Optional, Text, Tuple, Type
from tqdm import tqdm
from pathlib import Path

from rasa.constants import DOCS_URL_COMPONENTS
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    TEXT,
    TOKENS_NAMES,
    DENSE_FEATURE_NAMES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
)
import numpy as np
import tensorflow as tf

import rasa.utils.io as io_utils
import rasa.utils.train_utils as train_utils
import rasa.utils.common as common_utils

import os
import pickle
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
from .tokenizer import MecabTokenizer

from .network import FlairEmbedding

logger = logging.getLogger(__name__)

## custom constant
MODEL = 'model'
MODEL_PATH = 'model_path'
POS_NAME = 'pos'
MODEL_SIZE = 'model_size'
WINDOW_SIZE = 'window_size'
MIN_COUNT = 'min_count'
EPOCHS = 'epochs'
SEQ_LEN = 'seq_len'

class WordEmbedFeaturizer(DenseFeaturizer):
    """Featurizer using Word2Vec/FastText model.
    Loads the model and computes sentence and sequence level feature representations 
    for dense featurizable attributes of each message object.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [MecabTokenizer]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None, model = None) -> None:
        super(WordEmbedFeaturizer, self).__init__(component_config)
        self.model = model

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["gensim"]

    def get_data_from_examples(self, examples: List[Message], attribute: Text = TEXT) -> List[List[str]]:
        list_of_tokens = [example.get(TOKENS_NAMES[attribute]) for example in examples]
        list_of_pos = [example.get(POS_NAME) for example in examples]        
        data = []
        for i, j in zip(list_of_tokens, list_of_pos):
            data.append([_i.text + '_' + _j for _i, _j in zip(i, j)])
        return data

    def _train(self, training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any) -> None:
        non_empty_examples = []
        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            non_empty_examples += list(filter(lambda x: x.get(attribute), training_data.training_examples))
        
        data = self.get_data_from_examples(non_empty_examples)
        model_class = FastText if self.component_config[MODEL] == 'fasttext' else Word2Vec
        model = model_class(        
                data,
                size=self.component_config[MODEL_SIZE], 
                window=self.component_config[WINDOW_SIZE], 
                min_count=self.component_config[MIN_COUNT],
                iter=self.component_config[EPOCHS])
        return model

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
    
    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        print('Dense Featurizer Training...')
        self.model = self._train(training_data, config, **kwargs)
        print('Dense Featurizer Training Completed')

        batch_size = 64

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
    
    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        gensim_model_path = os.path.join(model_dir, f'{file_name}.gensim_model')
        self.model.save(gensim_model_path)
        return {"file" : file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component = None,
        **kwargs: Any,
    ) -> "Featurizer":
        """Loads the trained model from the provided directory."""

        if not model_dir or not meta.get("file"):
            logger.debug(
                f"Failed to load model. "
                f"Maybe the path '{os.path.abspath(model_dir)}' doesn't exist?"
            )
            return cls(component_config=meta)

        model_class = FastText if meta['model'] == 'fasttext' else Word2Vec
        model_path = os.path.join(model_dir, f"{meta['file']}.{meta['model']}")
        model = model_class(model_path)
        return cls(component_config=meta, model=model)



class FlairFeaturizer(DenseFeaturizer):
    """Featurizer using Flair embedding model.
    Loads the model and computes sentence and sequence level feature representations 
    for dense featurizable attributes of each message object.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [MecabTokenizer]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None, model = None, vocab = None) -> None:
        super(FlairFeaturizer, self).__init__(component_config)
        self.model = model
        self.vocab = vocab

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    def get_data_from_examples(self, examples: List[Message], attribute: Text = TEXT, return_vocab: bool = False) -> List[List[str]]:
        list_of_tokens = [example.get(TOKENS_NAMES[attribute])[:-1] for example in examples] # without cls token
        
        data, vocab = [], []
        for sentence in list_of_tokens:
            sent = []
            for token in sentence:
                token_text = token.text
                sent += list(token_text)
                sent.append('[SEP]')
                if return_vocab:
                    vocab += list(token_text)

            data.append(sent)

        if return_vocab:
            vocab = list(set(vocab))
            vocab = ['[PAD]', '[SEP]', '[UNK]'] + vocab
            return data, vocab
        else:
            return data

    def _pad_sequence(self, seq):
        seq = seq[:self.component_config[SEQ_LEN]]
        seq = np.pad(seq, (0, self.component_config[SEQ_LEN] - len(seq)), 'constant')
        return seq

    def _preprocess_data(self, data, vocab):
        char_to_idx = {j:i for i,j in enumerate(vocab)}
        input_data = [[char_to_idx.get(char, char_to_idx['[UNK]']) for char in sentence] for sentence in data]
        forward_data = [sentence[1:] for sentence in input_data]
        backward_data = [[char_to_idx['[PAD]']] + sentence[:-1] for sentence in input_data]

        input_data = np.array([self._pad_sequence(i) for i in input_data]).astype(np.int32)
        forward_data = np.array([self._pad_sequence(i) for i in forward_data]).astype(np.int32)
        backward_data = np.array([self._pad_sequence(i) for i in backward_data]).astype(np.int32)
        return input_data, forward_data, backward_data

    def _train(self, training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any) -> None:
        non_empty_examples = []
        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            non_empty_examples += list(filter(lambda x: x.get(attribute), training_data.training_examples))
        
        data, vocab = self.get_data_from_examples(non_empty_examples, return_vocab=True)
        input_data, forward_data, backward_data = self._preprocess_data(data, vocab)

        model = FlairEmbedding(len(vocab), self.component_config[MODEL_SIZE])
        model.compile(
            optimizer = tf.keras.optimizers.Adam(),
            loss = 'sparse_categorical_crossentropy'
        )
        model.fit(input_data, [forward_data, backward_data], epochs=self.component_config[EPOCHS])
        return model, vocab


    def get_representation(self, x):
        x = self.model.embedding(x)
        forward = self.model.forward_lstm(x)
        backward = self.model.backward_lstm(x)
        return forward, backward

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
            sequence_encoding = np.concatenate([sequence_encoding, sentence_encoding], axis=0)
            final_embeddings.append(sequence_encoding)

        return np.array(final_embeddings)

    def _compute_features(self, batch_examples: List[Message], attribute: Text = TEXT) -> Tuple[np.ndarray, List[int]]:

        batch_data = self.get_data_from_examples(batch_examples, attribute)
        batch_data, _, _ = self._preprocess_data(batch_data, self.vocab)
        forward, backward = self.get_representation(batch_data)

        list_of_tokens = [example.get(TOKENS_NAMES[attribute])[:-1] for example in batch_examples] # without cls token
        start_idx = [[t.start for t in tokens] for tokens in list_of_tokens]
        end_idx = [[t.end for t in tokens] for tokens in list_of_tokens]

        forward_encodings = [tf.gather(r, i) for i, r in zip(end_idx, forward)]
        backward_encodings = [tf.gather(r, i) for i, r in zip(start_idx, backward)]
        sequence_encodings = [tf.concat([f, b], axis=-1) for f, b in zip(forward_encodings, backward_encodings)]

        sequence_encodings = [np.array(i) for i in sequence_encodings]
        sentence_encodings = [np.mean(i, axis=0, keepdims=True) for i in sequence_encodings] # mean pooling
        # sentence_encodings = [np.max(i, axis=0, keepdims=True) for i in sequence_encodings] # max pooling

        number_of_tokens_in_sentence = [len(sentence) - 1 for sentence in batch_data] # without cls token
        return self._combine_encodings(sentence_encodings, sequence_encodings, number_of_tokens_in_sentence)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        print('Dense Featurizer Training...')
        self.model, self.vocab = self._train(training_data, config, **kwargs)
        print('Dense Featurizer Training Completed')

        batch_size = 64
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
        if not self.model:
            self.load_model()
            
        features = self._compute_features([message])[0]
        message.set(
            DENSE_FEATURE_NAMES[TEXT],
            self._combine_with_existing_dense_features(
                message, features, DENSE_FEATURE_NAMES[TEXT]
            ),
        )

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        tf_model_path = os.path.join(model_dir, f'{file_name}.tf_model')
        vocab_path = os.path.join(model_dir, f'{file_name}.vocab.pkl')
        os.makedirs(tf_model_path)

        tf.keras.models.save_model(self.model, tf_model_path)        
        io_utils.pickle_dump(vocab_path, self.vocab)
        return {"file" : file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component = None,
        **kwargs: Any,
    ) -> "Featurizer":
        """Loads the trained model from the provided directory."""

        if not model_dir or not meta.get("file"):
            logger.debug(
                f"Failed to load model. "
                f"Maybe the path '{os.path.abspath(model_dir)}' doesn't exist?"
            )
            return cls(component_config=meta)

        file_name = meta.get("file")
        tf_model_path = os.path.join(model_dir, f'{file_name}.tf_model')
        vocab_path = os.path.join(model_dir, f'{file_name}.vocab.pkl')

        model = tf.keras.models.load_model(tf_model_path)
        vocab = io_utils.pickle_load(vocab_path)

        return cls(component_config=meta, model=model, vocab=vocab)