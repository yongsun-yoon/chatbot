import logging
from typing import Any, Dict, List, Optional, Text, Tuple, Type
from tqdm import tqdm
from pathlib import Path

from rasa.constants import DOCS_URL_COMPONENTS
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
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
import pandas as pd
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
MODEL_SIZE = 'model_size'
WINDOW_SIZE = 'window_size'
MIN_COUNT = 'min_count'
EPOCHS = 'epochs'
SEQ_LEN = 'seq_len'
BUCKET_SIZE = 'bucket_size'
USE_DATA = 'use_data'
EXTERNAL_DATA_PATH = 'external_data_path'

def load_data(data_path):
    data_path = os.path.join(os.path.dirname(__file__), '..', data_path)
    data = pd.read_csv(data_path)
    data['token'] = data['token'].apply(lambda x : '/'.split(x))
    data = data['token'].to_list()
    return data

class WordEmbedFeaturizer(DenseFeaturizer):
    """Featurizer using Word2Vec/FastText model.
    Loads the model and computes sentence and sequence level feature representations 
    for dense featurizable attributes of each message object.
    """

    defaults = {
        USE_DATA : 'internal', # internal / external / both
        EXTERNAL_DATA_PATH : None,
        MODEL_SIZE : 64,
        WINDOW_SIZE : 7,
        MIN_COUNT : 1,
        BUCKET_SIZE : 100,
        EPOCHS : 10
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None, model = None, hash_embedding = None) -> None:
        super(WordEmbedFeaturizer, self).__init__(component_config)
        self.model = model
        self.hash_embedding = hash_embedding

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["gensim"]
    
    @staticmethod
    def model_class(config):
        if config[MODEL] == 'word2vec':
            model_class = Word2Vec
        elif config[MODEL] == 'fasttext':
            model_class = FastText
        return model_class

    def get_data_from_examples(self, examples: List[Message], attribute: Text = TEXT) -> List[List[str]]:
        list_of_tokens = [example.get(TOKENS_NAMES[attribute]) for example in examples]
        data = [[t.text for t in tokens] for tokens in list_of_tokens]
        return data

    def _train(self, training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any) -> None:

        if self.component_config[USE_DATA] == 'internal':
            non_empty_examples = []
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                non_empty_examples += list(filter(lambda x: x.get(attribute), training_data.training_examples))
            data = self.get_data_from_examples(non_empty_examples)
        
        elif self.component_config[USE_DATA] == 'external':
            data = load_data(self.component_config[EXTERNAL_DATA_PATH])

        elif self.component_config[USE_DATA] == 'both':
            non_empty_examples = []
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                non_empty_examples += list(filter(lambda x: x.get(attribute), training_data.training_examples))
            internal_data = self.get_data_from_examples(non_empty_examples)
            external_data = load_data(self.component_config[EXTERNAL_DATA_PATH])
            data = internal_data + external_data

        model_class = self.model_class(self.component_config)
        model = model_class(        
                data,
                size=self.component_config[MODEL_SIZE], 
                window=self.component_config[WINDOW_SIZE], 
                min_count=self.component_config[MIN_COUNT],
                iter=self.component_config[EPOCHS])

        hash_embedding = np.random.rand(self.component_config[BUCKET_SIZE], model.wv.vector_size)
        return model, hash_embedding

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
        
        sequence_encodings = []
        for sentence in batch_data:
            _sequence_encoding = []
            for token in sentence:
                if token in self.model.wv.vocab:
                    enc = self.model.wv[token]
                else:
                    enc = self.hash_embedding[tf.strings.to_hash_bucket(token, self.component_config[BUCKET_SIZE])]
                _sequence_encoding.append(enc)
            _sequence_encoding = np.array(_sequence_encoding)
            sequence_encodings.append(_sequence_encoding)

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
        self.model, self.hash_embedding = self._train(training_data, config, **kwargs)
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
        gensim_model_path = os.path.join(model_dir, f'{file_name}.model')
        hash_embedding_path = os.path.join(model_dir, f'{file_name}.hash_embedding.pkl')
        self.model.save(gensim_model_path)
        io_utils.pickle_dump(hash_embedding_path, self.hash_embedding)
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

        file_name = meta['file']
        gensim_model_path = os.path.join(model_dir, f"{file_name}.model")
        hash_embedding_path = os.path.join(model_dir, f'{file_name}.hash_embedding.pkl')

        model_class = cls.model_class(meta)
        model = model_class.load(gensim_model_path)
        hash_embedding = io_utils.pickle_load(hash_embedding_path)

        return cls(component_config=meta, model=model, hash_embedding=hash_embedding)



class FlairFeaturizer(DenseFeaturizer):
    """Featurizer using Flair embedding model.
    Loads the model and computes sentence and sequence level feature representations 
    for dense featurizable attributes of each message object.
    """
    defaults = {
        USE_DATA : 'internal',
        EXTERNAL_DATA_PATH : None,
        MODEL_SIZE : 64,
        SEQ_LEN : 100,
        EPOCHS : 100,
    }


    def __init__(self, component_config: Optional[Dict[Text, Any]] = None, model = None, vocab = None) -> None:
        super(FlairFeaturizer, self).__init__(component_config)
        self.model = model
        self.vocab = vocab

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    def get_data_from_external_data(self, external_data, return_vocab=False):
        data, vocab = [], []
        for sentence in external_data:
            sent = []
            sent.append('[BOS]')

            for token in sentence:
                sent += list(token)
                sent.append('[SEP]')

                if return_vocab:
                    vocab += list(token)
            
            data.append(sent)

        if return_vocab:
            vocab = list(set(vocab))
            vocab = ['[PAD]', '[SEP]', '[UNK]', '[BOS]'] + vocab
            return data, vocab
        else:
            return data


    def get_data_from_examples(self, examples: List[Message], attribute: Text = TEXT, return_vocab: bool = False) -> List[List[str]]:
        list_of_tokens = [example.get(TOKENS_NAMES[attribute])[:-1] for example in examples] # without cls token
        
        data, vocab = [], []
        for sentence in list_of_tokens:
            sent = []
            sent.append('[BOS]')
            add_idx = 1

            for token in sentence:
                token_text = token.text
                sent += list(token_text)
                token.char_start = token.start + add_idx
                token.char_end = token.end + add_idx
                
                sent.append('[SEP]')
                add_idx += 1
                if return_vocab:
                    vocab += list(token_text)
            data.append(sent)

        if return_vocab:
            vocab = list(set(vocab))
            vocab = ['[PAD]', '[SEP]', '[UNK]', '[BOS]'] + vocab
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
        
        if self.component_config[USE_DATA] == 'internal':
            non_empty_examples = []
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                non_empty_examples += list(filter(lambda x: x.get(attribute), training_data.training_examples))
            data, vocab = self.get_data_from_examples(non_empty_examples, return_vocab=True)
        
        elif self.component_config[USE_DATA] == 'external':
            data = load_data(self.component_config[EXTERNAL_DATA_PATH])
            data, vocab = self.get_data_from_external_data(data, return_vocab=True)

        elif self.component_config[USE_DATA] == 'both':
            non_empty_examples = []
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                non_empty_examples += list(filter(lambda x: x.get(attribute), training_data.training_examples))
            internal_data, internal_vocab = self.get_data_from_examples(non_empty_examples, return_vocab=True)
        
            external_data = load_data(self.component_config[EXTERNAL_DATA_PATH])
            external_data, external_vocab = self.get_data_from_external_data(external_data, return_vocab=True)

            data = internal_data + external_data
            vocab = list(set(internal_vocab + external_vocab))
        
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
        start_idx = [[t.char_start - 1 for t in tokens] for tokens in list_of_tokens]
        end_idx = [[t.char_end + 1 for t in tokens] for tokens in list_of_tokens]

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