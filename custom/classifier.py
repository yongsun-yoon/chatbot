import os
import logging
from typing import Any, Dict, Optional, Text, List, Type, Tuple

import rasa.utils.io as io_utils
from rasa.utils import train_utils
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.components import Component
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.training_data import Message, TrainingData
from rasa.utils.tensorflow.constants import (
    NUM_TRANSFORMER_LAYERS,
    TRANSFORMER_SIZE,
    NUM_HEADS,
    BATCH_SIZES,
    EPOCHS,
    RANDOM_SEED,
    RANKING_LENGTH,
    LEARNING_RATE,
    DROP_RATE,
)

from rasa.nlu.constants import (
    INTENT,
    TEXT,
    TOKENS_NAMES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
)

import rasa.utils.common as common_utils
from rasa.nlu.model import Metadata
from rasa.utils.tensorflow.models import RasaModel
from rasa.nlu.config import RasaNLUModelConfig, InvalidConfigError

import numpy as np
import tensorflow as tf
from .network import CharNetwork

logger = logging.getLogger(__name__)

SEQ_LEN = 'seq_len'

class CustomClassifier(IntentClassifier):
    """Character level Transformer used for intent classification.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Featurizer]

    # please make sure to update the docs when changing a default parameter
    defaults = {
        TRANSFORMER_SIZE: 512,
        NUM_HEADS: 4,
        NUM_TRANSFORMER_LAYERS: 2,
        DROP_RATE: 0.2,
        EPOCHS: 300,
        RANDOM_SEED: None,
        LEARNING_RATE: 0.001,
        SEQ_LEN : 100,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        index_label_id_mapping: Optional[Dict[int, Text]] = None,
        index_tag_id_mapping: Optional[Dict[int, Text]] = None,
        model = None,
        vocab = None,
    ) -> None:

        super().__init__(component_config)

        self.index_label_id_mapping = index_label_id_mapping
        self.index_tag_id_mapping = index_tag_id_mapping

        self.model = model
        self.vocab = vocab

    def prepare_label_dict(self, training_data: TrainingData, attribute: Text):
        """Create label_id dictionary."""
        distinct_label_ids = {example.get(attribute) for example in training_data.intent_examples} - {None}
        self.label_id_index_mapping = {label_id: idx for idx, label_id in enumerate(sorted(distinct_label_ids))} # str -> int
        self.index_label_id_mapping = {j:i for i,j in self.label_id_index_mapping.items()} # int -> str
        self._num_intent = len(self.index_label_id_mapping) if self.index_label_id_mapping is not None else 0


    def get_data_from_examples(self, examples: List[Message], attribute: Text = TEXT, return_vocab: bool = False) -> List[List[str]]:
        list_of_tokens = [example.get(TOKENS_NAMES[attribute])[:-1] for example in examples] # without cls token
        
        data, segment, vocab = [], [], []
        for sentence in list_of_tokens:
            sent, sent_segment = [], []
            seg_idx = 1
            sent.append('[BOS]')
            sent_segment.append(seg_idx)
            seg_idx += 1

            for token in sentence:
                token_text = token.text
                sent += list(token_text)
                sent.append('[SEP]')

                sent_segment += [seg_idx for _ in range(len(token_text) + 1)]
                seg_idx += 1

                if return_vocab:
                    vocab += list(token_text)
            
            data.append(sent)
            segment.append(sent_segment)

        if return_vocab:
            vocab = list(set(vocab))
            vocab = ['[PAD]', '[SEP]', '[UNK]', '[BOS]'] + vocab
            return data, segment, vocab
        else:
            return data, segment

    def _pad_sequence(self, seq):
        seq = seq[:self.component_config[SEQ_LEN]]
        seq = np.pad(seq, (0, self.component_config[SEQ_LEN] - len(seq)), 'constant')
        return seq

    def _get_input_data(self, data, segment, vocab):
        char_to_idx = {j:i for i,j in enumerate(vocab)}
        input_data = [[char_to_idx.get(char, char_to_idx['[UNK]']) for char in sentence] for sentence in data]
        input_data = np.array([self._pad_sequence(i) for i in input_data]).astype(np.int32)
        segment = np.array([self._pad_sequence(i) for i in segment]).astype(np.int32)
        return input_data, segment
    
    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        self.prepare_label_dict(training_data, attribute=INTENT)
        non_empty_examples, labels = [], []
        for e in training_data.training_examples:
            if e.get(TEXT):
                non_empty_examples.append(e)
                labels.append(self.label_id_index_mapping[e.get(INTENT)])

        data, segment, self.vocab = self.get_data_from_examples(non_empty_examples, return_vocab=True)
        input_data, segment = self._get_input_data(data, segment, self.vocab)
        labels = np.array(labels).astype(np.int32)

        print(input_data.shape)
        print(segment.shape)

        self.model = CharNetwork(
            num_intent = self._num_intent,
            vocab_size = len(self.vocab),
            model_dim=self.component_config[TRANSFORMER_SIZE], 
            ffn_dim=self.component_config[TRANSFORMER_SIZE], 
            num_head=self.component_config[NUM_HEADS], 
            drop_rate=self.component_config[DROP_RATE], 
            num_layer=self.component_config[NUM_TRANSFORMER_LAYERS])
        
        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(self.component_config[LEARNING_RATE]),
            loss = 'sparse_categorical_crossentropy'
        )
        
        self.model.fit([input_data, segment], labels, epochs=self.component_config[EPOCHS])

    # process helpers
    def _predict(self, message: Message) -> Optional[Dict[Text, tf.Tensor]]:
        if self.model is None:
            logger.debug(
                "There is no trained model: component is either not trained or "
                "didn't receive enough training data."
            )
            return

        # create session data from message and convert it into a batch of 1
        tokens = message.get(TOKENS_NAMES[TEXT])[:-1]
        data, segment = [], []
        seg_idx = 1

        data.append(['[BOS]'])
        segment.append(seg_idx)
        seg_idx += 1

        for token in tokens:
            token_text = token.text
            data += list(token_text)
            data.append('[SEP]')
            segment += [seg_idx for _ in range(len(token_text) + 1)]
            seg_idx += 1
        
        data = np.array(data)[None, :]
        return self.model.predict(data)

    def _predict_label(
        self, prob=None
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """Predicts the intent of the provided message."""

        label = {"name": None, "confidence": 0.0}
        label_ranking = []

        if prob is None:
            return label, label_ranking

        message_score = prob.numpy().flatten()
        label_ids = message_score.argsort()[::-1]

        message_score = train_utils.normalize(message_score, self.component_config[RANKING_LENGTH])
        message_score[::-1].sort()
        message_score = message_score.tolist()

        # if X contains all zeros do not predict some label
        if label_ids.size > 0:
            label = {
                "name": self.index_label_id_mapping[label_ids[0]],
                "confidence": message_score[0],
            }

            if (
                self.component_config[RANKING_LENGTH]
                and 0 < self.component_config[RANKING_LENGTH] < LABEL_RANKING_LENGTH
            ):
                output_length = self.component_config[RANKING_LENGTH]
            else:
                output_length = LABEL_RANKING_LENGTH

            ranking = list(zip(list(label_ids), message_score))
            ranking = ranking[:output_length]
            label_ranking = [
                {"name": self.index_label_id_mapping[label_idx], "confidence": score}
                for label_idx, score in ranking
            ]

        return label, label_ranking


    def process(self, message: Message, **kwargs: Any) -> None:
        """Return the most likely label and its similarity to the input."""

        prob = self._predict(message)
        label, label_ranking = self._predict_label(prob)

        message.set(INTENT, label, add_to_output=True)
        message.set("intent_ranking", label_ranking, add_to_output=True)


    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        tf_model_path = os.path.join(model_dir, f'{file_name}.tf_model')
        os.makedirs(tf_model_path)
        tf.keras.models.save_model(self.model, tf_model_path)

        io_utils.pickle_dump(os.path.join(model_dir, f'{file_name}.vocab.pkl'), self.vocab)
        io_utils.pickle_dump(os.path.join(model_dir, f'{file_name}.index_label_id_mapping.pkl'), self.index_label_id_mapping)
        io_utils.pickle_dump(os.path.join(model_dir, f'{file_name}.index_tag_id_mapping .pkl'), self.index_tag_id_mapping )
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
        model = tf.keras.models.load_model(tf_model_path)

        

        vocab = io_utils.pickle_load(os.path.join(model_dir, f'{file_name}.vocab.pkl'))
        index_label_id_mapping = io_utils.pickle_load(os.path.join(model_dir, f'{file_name}.index_label_id_mapping.pkl'))
        index_tag_id_mapping = io_utils.pickle_load(os.path.join(model_dir, f'{file_name}.index_tag_id_mapping .pkl'))

        return cls(
            component_config=meta, 
            index_label_id_mapping = index_label_id_mapping,
            index_tag_id_mapping = index_tag_id_mapping,
            model=model, 
            vocab=vocab)