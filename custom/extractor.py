import logging
from pathlib import Path

import numpy as np
import os
from collections import defaultdict
import scipy.sparse
import tensorflow as tf
import tensorflow_addons as tfa

from typing import Any, Dict, List, Optional, Text, Tuple, Union, Type

import rasa.utils.io as io_utils
import rasa.nlu.utils.bilou_utils as bilou_utils
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.components import Component
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.test import determine_token_labels
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.utils import train_utils
from rasa.utils.common import raise_warning
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.transformer import TransformerEncoder
from rasa.utils.tensorflow.models import RasaModel
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureSignature
from rasa.nlu.constants import (
    INTENT,
    TEXT,
    ENTITIES,
    NO_ENTITY_TAG,
    SPARSE_FEATURE_NAMES,
    DENSE_FEATURE_NAMES,
    TOKENS_NAMES,
)
from rasa.nlu.config import RasaNLUModelConfig, InvalidConfigError
from rasa.nlu.training_data import TrainingData
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message
from rasa.utils.tensorflow.constants import (
    LABEL,
    HIDDEN_LAYERS_SIZES,
    SHARE_HIDDEN_LAYERS,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    LEARNING_RATE,
    DENSE_DIMENSION,
    RANKING_LENGTH,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    SPARSE_INPUT_DROPOUT,
    MASKED_LM,
    ENTITY_RECOGNITION,
    TENSORBOARD_LOG_DIR,
    INTENT_CLASSIFICATION,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    UNIDIRECTIONAL_ENCODER,
    DROP_RATE,
    DROP_RATE_ATTENTION,
    WEIGHT_SPARSITY,
    NEGATIVE_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    BILOU_FLAG,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    SOFTMAX,
    AUTO,
    BALANCED,
    TENSORBOARD_LOG_LEVEL,
)

from .network import InputLayer, BaseLayer, IntentLayer, CRFEntityLayer, SoftmaxEntityLayer


logger = logging.getLogger(__name__)

TEXT_FEATURES = f"{TEXT}_features"
LABEL_FEATURES = f"{LABEL}_features"
TEXT_MASK = f"{TEXT}_mask"
LABEL_MASK = f"{LABEL}_mask"
LABEL_IDS = f"{LABEL}_ids"
TAG_IDS = "tag_ids"
MODEL = 'model'
DICE_GAMMA = 'dice_gamma'


class CustomExtractor(IntentClassifier, EntityExtractor):
    """DIET (Dual Intent and Entity Transformer) is a multi-task architecture for
    intent classification and entity recognition.

    The architecture is based on a transformer which is shared for both tasks.
    A sequence of entity labels is predicted through a Conditional Random Field (CRF)
    tagging layer on top of the transformer output sequence corresponding to the
    input sequence of tokens. The transformer output for the ``__CLS__`` token and
    intent labels are embedded into a single semantic vector space. We use the
    dot-product loss to maximize the similarity with the target label and minimize
    similarities with negative samples.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Featurizer]

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # ## Basic parameters
        INTENT_CLASSIFICATION: True,
        ENTITY_RECOGNITION: True,
        MASKED_LM: False,
        BILOU_FLAG: True,
        TENSORBOARD_LOG_DIR: None,
        TENSORBOARD_LOG_LEVEL: "epoch",

        # ## Model parameters
        TRANSFORMER_SIZE: 256,
        NUM_TRANSFORMER_LAYERS: 2,
        NUM_HEADS: 4,
        EMBEDDING_DIMENSION: [64, 128],  # Dimension size of embedding vectors
        REGULARIZATION_CONSTANT: 0.002,
        DROP_RATE: 0.2,
        WEIGHT_SPARSITY: 0.8,
        SPARSE_INPUT_DROPOUT: True,
        DICE_GAMMA: 0.2,

        # ## Training parameters
        EPOCHS: 100,
        RANDOM_SEED: None,
        LEARNING_RATE: 0.001,
        BATCH_SIZES: [64, 256], # Batch size will be linearly increased for each epoch.
        BATCH_STRATEGY: BALANCED, # Strategy used when creating batches. ('sequence' or 'balanced')

        # ## Evaluation parameters
        EVAL_NUM_EPOCHS: 20,
        EVAL_NUM_EXAMPLES: 0,
        RANKING_LENGTH: 20,
        
    }


    def _check_config_parameters(self) -> None:
        self.component_config = train_utils.check_deprecated_options(
            self.component_config
        )

        self.component_config = train_utils.update_similarity_type(
            self.component_config
        )
        self.component_config = train_utils.update_evaluation_parameters(
            self.component_config
        )

    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        index_label_id_mapping: Optional[Dict[int, Text]] = None,
        index_tag_id_mapping: Optional[Dict[int, Text]] = None,
        model: Optional[RasaModel] = None,
    ) -> None:
        """Declare instance variables with default values."""

        if component_config is not None and EPOCHS not in component_config:
            raise_warning(
                f"Please configure the number of '{EPOCHS}' in your configuration file."
                f" We will change the default value of '{EPOCHS}' in the future to 1. "
            )

        super().__init__(component_config)

        self._check_config_parameters()

        # transform numbers to labels
        self.index_label_id_mapping = index_label_id_mapping
        self.index_tag_id_mapping = index_tag_id_mapping

        self.model = model

        self.num_tags: Optional[int] = None  # number of entity tags
        self._label_data: Optional[RasaModelData] = None
        self.data_example: Optional[Dict[Text, List[np.ndarray]]] = None

    @property
    def label_key(self) -> Optional[Text]:
        return LABEL_IDS if self.component_config[INTENT_CLASSIFICATION] else None

    @staticmethod
    def model_class(config) -> Type[RasaModel]:
        if config[MODEL] == 'CRF':
            model_class = CRFTransformer
        elif config[MODEL] == 'Dice':
            model_class = DiceTransformer
        return model_class

    # training data helpers:
    @staticmethod
    def _label_id_index_mapping(
        training_data: TrainingData, attribute: Text
    ) -> Dict[Text, int]:
        """Create label_id dictionary."""

        distinct_label_ids = {
            example.get(attribute) for example in training_data.intent_examples
        } - {None}
        return {
            label_id: idx for idx, label_id in enumerate(sorted(distinct_label_ids))
        }

    @staticmethod
    def _invert_mapping(mapping: Dict) -> Dict:
        return {value: key for key, value in mapping.items()}

    def _tag_id_index_mapping(self, training_data: TrainingData) -> Dict[Text, int]:
        """Create tag_id dictionary"""

        if self.component_config[BILOU_FLAG]:
            return bilou_utils.build_tag_id_dict(training_data)

        distinct_tag_ids = set(
            e["entity"]
            for example in training_data.entity_examples
            for e in example.get(ENTITIES)
        ) - {None}

        tag_id_dict = {
            tag_id: idx for idx, tag_id in enumerate(sorted(distinct_tag_ids), 1)
        }
        # NO_ENTITY_TAG corresponds to non-entity which should correspond to 0 index
        # needed for correct prediction for padding
        tag_id_dict[NO_ENTITY_TAG] = 0

        return tag_id_dict

    @staticmethod
    def _find_example_for_label(
        label: Text, examples: List[Message], attribute: Text
    ) -> Optional[Message]:
        for ex in examples:
            if ex.get(attribute) == label:
                return ex
        return None

    @staticmethod
    def _check_labels_features_exist(
        labels_example: List[Message], attribute: Text
    ) -> bool:
        """Checks if all labels have features set."""

        return all(
            label_example.get(SPARSE_FEATURE_NAMES[attribute]) is not None
            or label_example.get(DENSE_FEATURE_NAMES[attribute]) is not None
            for label_example in labels_example
        )

    def _extract_features(
        self, message: Message, attribute: Text
    ) -> Tuple[Optional[scipy.sparse.spmatrix], Optional[np.ndarray]]:
        sparse_features = None
        dense_features = None

        if message.get(SPARSE_FEATURE_NAMES[attribute]) is not None:
            sparse_features = message.get(SPARSE_FEATURE_NAMES[attribute])

        if message.get(DENSE_FEATURE_NAMES[attribute]) is not None:
            dense_features = message.get(DENSE_FEATURE_NAMES[attribute])

        if sparse_features is not None and dense_features is not None:
            if sparse_features.shape[0] != dense_features.shape[0]:
                raise ValueError(
                    f"Sequence dimensions for sparse and dense features "
                    f"don't coincide in '{message.text}' for attribute '{attribute}'."
                )

        # If we don't use the transformer and we don't want to do entity recognition,
        # to speed up training take only the sentence features as feature vector.
        # It corresponds to the feature vector for the last token - CLS token.
        # We would not make use of the sequence anyway in this setup. Carrying over
        # those features to the actual training process takes quite some time.
        if (
            self.component_config[NUM_TRANSFORMER_LAYERS] == 0
            and not self.component_config[ENTITY_RECOGNITION]
            and attribute != INTENT
        ):
            sparse_features = train_utils.sequence_to_sentence_features(sparse_features)
            dense_features = train_utils.sequence_to_sentence_features(dense_features)

        return sparse_features, dense_features

    def _check_input_dimension_consistency(self, model_data: RasaModelData) -> None:
        """Checks if features have same dimensionality if hidden layers are shared."""

        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            num_text_features = model_data.feature_dimension(TEXT_FEATURES)
            num_label_features = model_data.feature_dimension(LABEL_FEATURES)

            if num_text_features != num_label_features:
                raise ValueError(
                    "If embeddings are shared text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )

    def _extract_labels_precomputed_features(
        self, label_examples: List[Message], attribute: Text = INTENT
    ) -> List[np.ndarray]:
        """Collects precomputed encodings."""

        sparse_features = []
        dense_features = []

        for e in label_examples:
            _sparse, _dense = self._extract_features(e, attribute)
            if _sparse is not None:
                sparse_features.append(_sparse)
            if _dense is not None:
                dense_features.append(_dense)

        sparse_features = np.array(sparse_features)
        dense_features = np.array(dense_features)

        return [sparse_features, dense_features]

    @staticmethod
    def _compute_default_label_features(
        labels_example: List[Message],
    ) -> List[np.ndarray]:
        """Computes one-hot representation for the labels."""

        eye_matrix = np.eye(len(labels_example), dtype=np.float32)
        # add sequence dimension to one-hot labels
        return [np.array([np.expand_dims(a, 0) for a in eye_matrix])]

    def _create_label_data(
        self,
        training_data: TrainingData,
        label_id_dict: Dict[Text, int],
        attribute: Text,
    ) -> RasaModelData:
        """Create matrix with label_ids encoded in rows as bag of words.

        Find a training example for each label and get the encoded features
        from the corresponding Message object.
        If the features are already computed, fetch them from the message object
        else compute a one hot encoding for the label as the feature vector.
        """

        # Collect one example for each label
        labels_idx_examples = []
        for label_name, idx in label_id_dict.items():
            label_example = self._find_example_for_label(
                label_name, training_data.intent_examples, attribute
            )
            labels_idx_examples.append((idx, label_example))

        # Sort the list of tuples based on label_idx
        labels_idx_examples = sorted(labels_idx_examples, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_examples]

        # Collect features, precomputed if they exist, else compute on the fly
        if self._check_labels_features_exist(labels_example, attribute):
            features = self._extract_labels_precomputed_features(
                labels_example, attribute
            )
        else:
            features = self._compute_default_label_features(labels_example)

        label_data = RasaModelData()
        label_data.add_features(LABEL_FEATURES, features)

        label_ids = np.array([idx for (idx, _) in labels_idx_examples])
        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        label_data.add_features(LABEL_IDS, [np.expand_dims(label_ids, -1)])

        label_data.add_mask(LABEL_MASK, LABEL_FEATURES)

        return label_data

    def _use_default_label_features(self, label_ids: np.ndarray) -> List[np.ndarray]:
        all_label_features = self._label_data.get(LABEL_FEATURES)[0]
        return [np.array([all_label_features[label_id] for label_id in label_ids])]

    def _create_model_data(
        self,
        training_data: List[Message],
        label_id_dict: Optional[Dict[Text, int]] = None,
        tag_id_dict: Optional[Dict[Text, int]] = None,
        label_attribute: Optional[Text] = None,
    ) -> RasaModelData:
        """Prepare data for training and create a RasaModelData object"""

        X_sparse = []
        X_dense = []
        Y_sparse = []
        Y_dense = []
        label_ids = []
        tag_ids = []

        for e in training_data:
            if label_attribute is None or e.get(label_attribute):
                _sparse, _dense = self._extract_features(e, TEXT)
                if _sparse is not None:
                    X_sparse.append(_sparse)
                if _dense is not None:
                    X_dense.append(_dense)

            if e.get(label_attribute):
                _sparse, _dense = self._extract_features(e, label_attribute)
                if _sparse is not None:
                    Y_sparse.append(_sparse)
                if _dense is not None:
                    Y_dense.append(_dense)

                if label_id_dict:
                    label_ids.append(label_id_dict[e.get(label_attribute)])

            if self.component_config.get(ENTITY_RECOGNITION) and tag_id_dict:
                if self.component_config[BILOU_FLAG]:
                    _tags = bilou_utils.tags_to_ids(e, tag_id_dict)
                else:
                    _tags = []
                    for t in e.get(TOKENS_NAMES[TEXT]):
                        _tag = determine_token_labels(t, e.get(ENTITIES), None)
                        _tags.append(tag_id_dict[_tag])
                # transpose to have seq_len x 1
                tag_ids.append(np.array([_tags]).T)

        X_sparse = np.array(X_sparse)
        X_dense = np.array(X_dense)
        Y_sparse = np.array(Y_sparse)
        Y_dense = np.array(Y_dense)
        label_ids = np.array(label_ids)
        tag_ids = np.array(tag_ids)

        model_data = RasaModelData(label_key=self.label_key)
        model_data.add_features(TEXT_FEATURES, [X_sparse, X_dense])
        model_data.add_features(LABEL_FEATURES, [Y_sparse, Y_dense])
        if label_attribute and model_data.feature_not_exist(LABEL_FEATURES):
            # no label features are present, get default features from _label_data
            model_data.add_features(
                LABEL_FEATURES, self._use_default_label_features(label_ids)
            )

        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        model_data.add_features(LABEL_IDS, [np.expand_dims(label_ids, -1)])
        model_data.add_features(TAG_IDS, [tag_ids])

        model_data.add_mask(TEXT_MASK, TEXT_FEATURES)
        model_data.add_mask(LABEL_MASK, LABEL_FEATURES)

        return model_data

    # train helpers
    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.
        """

        if self.component_config[BILOU_FLAG]:
            bilou_utils.apply_bilou_schema(training_data)

        label_id_index_mapping = self._label_id_index_mapping(
            training_data, attribute=INTENT
        )

        if not label_id_index_mapping:
            # no labels are present to train
            return RasaModelData()

        self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)

        self._label_data = self._create_label_data(
            training_data, label_id_index_mapping, attribute=INTENT
        )

        tag_id_index_mapping = self._tag_id_index_mapping(training_data)
        self.index_tag_id_mapping = self._invert_mapping(tag_id_index_mapping)

        label_attribute = (
            INTENT if self.component_config[INTENT_CLASSIFICATION] else None
        )

        model_data = self._create_model_data(
            training_data.training_examples,
            label_id_index_mapping,
            tag_id_index_mapping,
            label_attribute=label_attribute,
        )

        self.num_tags = len(self.index_tag_id_mapping)

        self._check_input_dimension_consistency(model_data)

        return model_data

    @staticmethod
    def _check_enough_labels(model_data: RasaModelData) -> bool:
        return len(np.unique(model_data.get(LABEL_IDS))) >= 2

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train the embedding intent classifier on a data set."""

        model_data = self.preprocess_train_data(training_data)
        if model_data.is_empty():
            logger.debug(
                f"Cannot train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the classifier."
            )
            return

        if self.component_config.get(INTENT_CLASSIFICATION):
            if not self._check_enough_labels(model_data):
                logger.error(
                    f"Cannot train '{self.__class__.__name__}'. "
                    f"Need at least 2 different intent classes. "
                    f"Skipping training of classifier."
                )
                return

        # keep one example for persisting and loading
        self.data_example = model_data.first_data_example()

        self.model = self.model_class(self.component_config)(
            data_signature=model_data.get_signature(),
            label_data=self._label_data,
            index_label_id_mapping=self.index_label_id_mapping,
            index_tag_id_mapping=self.index_tag_id_mapping,
            config=self.component_config,
        )

        self.model.fit(
            model_data,
            self.component_config[EPOCHS],
            self.component_config[BATCH_SIZES],
            self.component_config[EVAL_NUM_EXAMPLES],
            self.component_config[EVAL_NUM_EPOCHS],
            self.component_config[BATCH_STRATEGY],
        )

    # process helpers
    def _predict(self, message: Message) -> Optional[Dict[Text, tf.Tensor]]:
        if self.model is None:
            logger.debug(
                "There is no trained model: component is either not trained or "
                "didn't receive enough training data."
            )
            return

        # create session data from message and convert it into a batch of 1
        model_data = self._create_model_data([message])

        return self.model.predict(model_data)

    def _predict_label(
        self, predict_out: Optional[Dict[Text, tf.Tensor]]
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """Predicts the intent of the provided message."""

        label = {"name": None, "confidence": 0.0}
        label_ranking = []

        if predict_out is None:
            return label, label_ranking

        message_score = predict_out["i_scores"].numpy().flatten()
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

    def _predict_entities(
        self, predict_out: Optional[Dict[Text, tf.Tensor]], message: Message
    ) -> List[Dict]:
        if predict_out is None:
            return []

        # load tf graph and session
        predictions = predict_out["e_ids"].numpy()

        tags = [self.index_tag_id_mapping[p] for p in predictions[0]]

        if self.component_config[BILOU_FLAG]:
            tags = bilou_utils.remove_bilou_prefixes(tags)

        entities = self._convert_tags_to_entities(
            message.text, message.get(TOKENS_NAMES[TEXT], []), tags
        )

        extracted = self.add_extractor_name(entities)
        entities = message.get(ENTITIES, []) + extracted

        return entities

    @staticmethod
    def _convert_tags_to_entities(
        text: Text, tokens: List[Token], tags: List[Text]
    ) -> List[Dict[Text, Any]]:
        entities = []
        last_tag = NO_ENTITY_TAG
        for token, tag in zip(tokens, tags):
            if tag == NO_ENTITY_TAG:
                last_tag = tag
                continue

            # new tag found
            if last_tag != tag:
                entity = {
                    "entity": tag,
                    "start": token.start,
                    "end": token.end,
                    "extractor": "DIET",
                }
                entities.append(entity)

            # belongs to last entity
            elif last_tag == tag:
                entities[-1]["end"] = token.end

            last_tag = tag

        for entity in entities:
            entity["value"] = text[entity["start"] : entity["end"]]

        return entities

    def process(self, message: Message, **kwargs: Any) -> None:
        """Return the most likely label and its similarity to the input."""

        out = self._predict(message)

        if self.component_config[INTENT_CLASSIFICATION]:
            label, label_ranking = self._predict_label(out)

            message.set(INTENT, label, add_to_output=True)
            message.set("intent_ranking", label_ranking, add_to_output=True)

        if self.component_config[ENTITY_RECOGNITION]:
            entities = self._predict_entities(out, message)

            message.set(ENTITIES, entities, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        if self.model is None:
            return {"file": None}

        model_dir = Path(model_dir)
        tf_model_file = model_dir / f"{file_name}.tf_model"

        io_utils.create_directory_for_file(tf_model_file)

        self.model.save(str(tf_model_file))

        io_utils.pickle_dump(
            model_dir / f"{file_name}.data_example.pkl", self.data_example
        )
        io_utils.pickle_dump(
            model_dir / f"{file_name}.label_data.pkl", self._label_data
        )
        io_utils.json_pickle(
            model_dir / f"{file_name}.index_label_id_mapping.pkl",
            self.index_label_id_mapping,
        )
        io_utils.json_pickle(
            model_dir / f"{file_name}.index_tag_id_mapping.pkl",
            self.index_tag_id_mapping,
        )

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["DIETClassifier"] = None,
        **kwargs: Any,
    ) -> "Classifier":
        """Loads the trained model from the provided directory."""

        if not model_dir or not meta.get("file"):
            logger.debug(
                f"Failed to load model. "
                f"Maybe the path '{os.path.abspath(model_dir)}' doesn't exist?"
            )
            return cls(component_config=meta)

        (
            index_label_id_mapping,
            index_tag_id_mapping,
            label_data,
            meta,
            data_example,
        ) = cls._load_from_files(meta, model_dir)

        meta = train_utils.update_similarity_type(meta)

        model = cls._load_model(
            index_label_id_mapping, index_tag_id_mapping, label_data, meta, data_example, model_dir
        )

        return cls(
            component_config=meta,
            index_label_id_mapping=index_label_id_mapping,
            index_tag_id_mapping=index_tag_id_mapping,
            model=model,
        )

    @classmethod
    def _load_from_files(cls, meta: Dict[Text, Any], model_dir: Text):
        file_name = meta.get("file")

        model_dir = Path(model_dir)

        data_example = io_utils.pickle_load(model_dir / f"{file_name}.data_example.pkl")
        label_data = io_utils.pickle_load(model_dir / f"{file_name}.label_data.pkl")
        index_label_id_mapping = io_utils.json_unpickle(
            model_dir / f"{file_name}.index_label_id_mapping.pkl"
        )
        index_tag_id_mapping = io_utils.json_unpickle(
            model_dir / f"{file_name}.index_tag_id_mapping.pkl"
        )

        # jsonpickle converts dictionary keys to strings
        index_label_id_mapping = {
            int(key): value for key, value in index_label_id_mapping.items()
        }
        if index_tag_id_mapping is not None:
            index_tag_id_mapping = {
                int(key): value for key, value in index_tag_id_mapping.items()
            }

        return (
            index_label_id_mapping,
            index_tag_id_mapping,
            label_data,
            meta,
            data_example,
        )

    @classmethod
    def _load_model(
        cls,
        index_label_id_mapping: Dict[int, Text],
        index_tag_id_mapping: Dict[int, Text],
        label_data: RasaModelData,
        meta: Dict[Text, Any],
        data_example: Dict[Text, List[np.ndarray]],
        model_dir: Text,
    ):
        file_name = meta.get("file")
        tf_model_file = os.path.join(model_dir, file_name + ".tf_model")

        label_key = LABEL_IDS if meta[INTENT_CLASSIFICATION] else None
        model_data_example = RasaModelData(label_key=label_key, data=data_example)

        model = cls.model_class(meta).load(
            tf_model_file,
            model_data_example,
            data_signature=model_data_example.get_signature(),
            label_data=label_data,
            index_label_id_mapping=index_label_id_mapping,
            index_tag_id_mapping=index_tag_id_mapping,
            config=meta,
        )

        # build the graph for prediction
        predict_data_example = RasaModelData(
            label_key=label_key,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if TEXT in feature_name
            },
        )

        model.build_for_predict(predict_data_example)

        return model


# accessing _tf_layers with any key results in key-error, disable it
# pytype: disable=key-error


class CRFTransformer(RasaModel):
    def __init__(
        self,
        data_signature: Dict[Text, List[FeatureSignature]],
        label_data: RasaModelData,
        index_label_id_mapping: Optional[Dict[int, Text]],
        index_tag_id_mapping: Optional[Dict[int, Text]],
        config: Dict[Text, Any],
    ) -> None:
        
        super().__init__(
            name="CRFTransformer",
            random_seed=config[RANDOM_SEED],
            tensorboard_log_dir=config[TENSORBOARD_LOG_DIR],
            tensorboard_log_level=config[TENSORBOARD_LOG_LEVEL],
        )

        self.config = config
        self.data_signature = data_signature
        self._check_data()

        self.predict_data_signature = {feature_name : features for feature_name, features in data_signature.items() if TEXT in feature_name}
        label_batch = label_data.prepare_batch()
        self.tf_label_data = self.batch_to_model_data_format(label_batch, label_data.get_signature())
        self._num_intents = len(index_label_id_mapping) if index_label_id_mapping is not None else 0
        self._num_tags = len(index_tag_id_mapping) if index_tag_id_mapping is not None else 0

        # tf objects, training
        self._prepare_layers()
        self._set_optimizer(tf.keras.optimizers.Adam(config[LEARNING_RATE]))
        self._create_metrics()
        self._update_metrics_to_log()

    def _check_data(self) -> None:
        if TEXT_FEATURES not in self.data_signature:
            raise InvalidConfigError(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if self.config[INTENT_CLASSIFICATION]:
            if LABEL_FEATURES not in self.data_signature:
                raise InvalidConfigError(
                    f"No label features specified. "
                    f"Cannot train '{self.__class__.__name__}' model."
                )

        if self.config[ENTITY_RECOGNITION] and TAG_IDS not in self.data_signature:
            raise ValueError(
                f"No tag ids present. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

    def _create_metrics(self) -> None:
        self.intent_loss = tf.keras.metrics.Mean(name="i_loss")
        self.entity_loss = tf.keras.metrics.Mean(name="e_loss")
        self.intent_acc = tf.keras.metrics.Mean(name="i_acc")
        self.entity_f1 = tf.keras.metrics.Mean(name="e_f1")

    def _update_metrics_to_log(self) -> None:
        if self.config[INTENT_CLASSIFICATION]:
            self.metrics_to_log += ["i_loss", "i_acc"]
        if self.config[ENTITY_RECOGNITION]:
            self.metrics_to_log += ["e_loss", "e_f1"]

    def _prepare_layers(self):
        self._tf_layers: Dict[Text : tf.keras.layers.Layer] = {}
        self._tf_layers['input_layer'] = InputLayer(dense_dim=self.config[EMBEDDING_DIMENSION], model_dim=self.config[TRANSFORMER_SIZE], reg_lambda=self.config[REGULARIZATION_CONSTANT], drop_rate=self.config[DROP_RATE])
        self._tf_layers['base_layer'] = BaseLayer(model_dim=self.config[TRANSFORMER_SIZE], ffn_dim=self.config[TRANSFORMER_SIZE], num_head=self.config[NUM_HEADS], drop_rate=self.config[DROP_RATE], num_layer=self.config[NUM_TRANSFORMER_LAYERS])
        self._tf_layers['intent_layer'] = IntentLayer(self._num_intents)
        self._tf_layers['entity_layer'] = CRFEntityLayer(self._num_tags)
        self._tf_layers['intent_acc'] = tf.metrics.Accuracy()
        self._tf_layers["entity_f1"] = tfa.metrics.F1Score(num_classes=self._num_tags - 1, average="micro")  # `0` prediction is not a prediction

    @staticmethod
    def _get_sequence_lengths(mask: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.reduce_sum(mask[:, :, 0], axis=1), tf.int32)

    def batch_loss(self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> tf.Tensor:

        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)
        mask = tf_batch_data[TEXT_MASK][0]
        attn_mask = mask[:, None, None, :, 0]
        sequence_lengths = self._get_sequence_lengths(mask)
        
        inputs = self._tf_layers[f"input_layer"](tf_batch_data[TEXT_FEATURES], mask, sparse_dropout=self.config[SPARSE_INPUT_DROPOUT], training=self._training)
        x = self._tf_layers[f"base_layer"](inputs, 1 - attn_mask, self._training)

        losses = []
        if self.config[INTENT_CLASSIFICATION]:
            preds = self._tf_layers['intent_layer'](x, sequence_lengths, training=self._training)
            labels = tf_batch_data[LABEL_IDS][0][:,0]
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, preds)
            loss = tf.reduce_mean(loss)
            acc = tf.keras.metrics.categorical_accuracy(labels, tf.argmax(preds, axis=-1))            
            
            losses.append(loss)
            self.intent_loss.update_state(loss)
            self.intent_acc.update_state(acc)

        if self.config[ENTITY_RECOGNITION]:
            labels = tf_batch_data[TAG_IDS][0]
            labels = tf.cast(labels[:, :, 0], tf.int32)
            log_likelihood, preds = self._tf_layers['entity_layer'](x, labels, sequence_lengths-1, training=self._training)
            loss = tf.reduce_mean(-log_likelihood)
            
            mask_bool = tf.cast(mask[:, :, 0], tf.bool)
            # pick only non padding values and flatten sequences
            labels_flat = tf.boolean_mask(labels, mask_bool)
            preds_flat = tf.boolean_mask(preds, mask_bool)
            # set `0` prediction to not a prediction
            labels_flat_one_hot = tf.one_hot(labels_flat - 1, self._num_tags - 1)
            preds_flat_one_hot = tf.one_hot(preds_flat - 1, self._num_tags - 1)
            
            losses.append(loss)
            entity_f1 = self._tf_layers['entity_f1'](labels_flat_one_hot, preds_flat_one_hot)
            self.entity_loss.update_state(loss)
            self.entity_f1.update_state(entity_f1)

        losses = tf.add_n(losses)
        return losses
    
    def batch_predict(self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, tf.Tensor]:
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.predict_data_signature)

        mask = tf_batch_data[TEXT_MASK][0]
        attn_mask = mask[:, None, None, :, 0]
        sequence_lengths = self._get_sequence_lengths(mask)
        
        inputs = self._tf_layers[f"input_layer"](tf_batch_data[TEXT_FEATURES], mask, sparse_dropout=self.config[SPARSE_INPUT_DROPOUT], training=self._training)
        x = self._tf_layers[f"base_layer"](inputs, 1 - attn_mask, self._training)
        
        out = {}
        if self.config[INTENT_CLASSIFICATION]:
            preds = self._tf_layers['intent_layer'](x, sequence_lengths, training=self._training)
            out['i_scores'] = preds
        
        if self.config[ENTITY_RECOGNITION]:
            labels = None
            _, preds = self._tf_layers['entity_layer'](x, labels, sequence_lengths-1, training=self._training)
            out['e_ids'] = preds
        return out


class DiceTransformer(RasaModel):
    def __init__(
        self,
        data_signature: Dict[Text, List[FeatureSignature]],
        label_data: RasaModelData,
        index_label_id_mapping: Optional[Dict[int, Text]],
        index_tag_id_mapping: Optional[Dict[int, Text]],
        config: Dict[Text, Any],
    ) -> None:
        
        super().__init__(
            name="DiceTransformer",
            random_seed=config[RANDOM_SEED],
            tensorboard_log_dir=config[TENSORBOARD_LOG_DIR],
            tensorboard_log_level=config[TENSORBOARD_LOG_LEVEL],
        )

        self.config = config
        self.data_signature = data_signature
        self._check_data()

        self.predict_data_signature = {feature_name : features for feature_name, features in data_signature.items() if TEXT in feature_name}
        label_batch = label_data.prepare_batch()
        self.tf_label_data = self.batch_to_model_data_format(label_batch, label_data.get_signature())
        self._num_intents = len(index_label_id_mapping) if index_label_id_mapping is not None else 0
        self._num_tags = len(index_tag_id_mapping) if index_tag_id_mapping is not None else 0

        # tf objects, training
        self._prepare_layers()
        self._set_optimizer(tf.keras.optimizers.Adam(config[LEARNING_RATE]))
        self._create_metrics()
        self._update_metrics_to_log()

    def _check_data(self) -> None:
        if TEXT_FEATURES not in self.data_signature:
            raise InvalidConfigError(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if self.config[INTENT_CLASSIFICATION]:
            if LABEL_FEATURES not in self.data_signature:
                raise InvalidConfigError(
                    f"No label features specified. "
                    f"Cannot train '{self.__class__.__name__}' model."
                )

        if self.config[ENTITY_RECOGNITION] and TAG_IDS not in self.data_signature:
            raise ValueError(
                f"No tag ids present. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

    def _create_metrics(self) -> None:
        self.intent_loss = tf.keras.metrics.Mean(name="i_loss")
        self.entity_loss = tf.keras.metrics.Mean(name="e_loss")
        self.intent_acc = tf.keras.metrics.Mean(name="i_acc")
        self.entity_f1 = tf.keras.metrics.Mean(name="e_f1")

    def _update_metrics_to_log(self) -> None:
        if self.config[INTENT_CLASSIFICATION]:
            self.metrics_to_log += ["i_loss", "i_acc"]
        if self.config[ENTITY_RECOGNITION]:
            self.metrics_to_log += ["e_loss", "e_f1"]

    def _prepare_layers(self):
        self._tf_layers: Dict[Text : tf.keras.layers.Layer] = {}
        self._tf_layers['input_layer'] = InputLayer(dense_dim=self.config[EMBEDDING_DIMENSION], model_dim=self.config[TRANSFORMER_SIZE], reg_lambda=self.config[REGULARIZATION_CONSTANT], drop_rate=self.config[DROP_RATE])
        self._tf_layers['base_layer'] = BaseLayer(model_dim=self.config[TRANSFORMER_SIZE], ffn_dim=self.config[TRANSFORMER_SIZE], num_head=self.config[NUM_HEADS], drop_rate=self.config[DROP_RATE], num_layer=self.config[NUM_TRANSFORMER_LAYERS])
        self._tf_layers['intent_layer'] = IntentLayer(self._num_intents)
        self._tf_layers['entity_layer'] = SoftmaxEntityLayer(self._num_tags)
        self._tf_layers['intent_acc'] = tf.metrics.Accuracy()
        self._tf_layers["entity_f1"] = tfa.metrics.F1Score(num_classes=self._num_tags - 1, average="micro")  # `0` prediction is not a prediction

    @staticmethod
    def _get_sequence_lengths(mask: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.reduce_sum(mask[:, :, 0], axis=1), tf.int32)

    def batch_loss(self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> tf.Tensor:

        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)
        mask = tf_batch_data[TEXT_MASK][0]
        attn_mask = mask[:, None, None, :, 0]
        sequence_lengths = self._get_sequence_lengths(mask)
        
        inputs = self._tf_layers[f"input_layer"](tf_batch_data[TEXT_FEATURES], mask, sparse_dropout=self.config[SPARSE_INPUT_DROPOUT], training=self._training)
        x = self._tf_layers[f"base_layer"](inputs, 1 - attn_mask, self._training)

        losses = []
        if self.config[INTENT_CLASSIFICATION]:
            preds = self._tf_layers['intent_layer'](x, sequence_lengths, training=self._training)
            labels = tf_batch_data[LABEL_IDS][0][:,0]
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, preds)
            loss = tf.reduce_mean(loss)
            acc = tf.keras.metrics.categorical_accuracy(labels, tf.argmax(preds, axis=-1))            
            
            losses.append(loss)
            self.intent_loss.update_state(loss)
            self.intent_acc.update_state(acc)

        if self.config[ENTITY_RECOGNITION]:
            labels = tf_batch_data[TAG_IDS][0]
            labels = tf.cast(labels[:, :, 0], tf.int32)
            labels_onehot = tf.one_hot(labels, depth=self._num_tags, axis=-1)
            labels_onehot = tf.cast(labels_onehot, tf.float32)
            prob, preds = self._tf_layers['entity_layer'](x, sequence_lengths-1, training=self._training)
            
            nom = 2 * prob * labels_onehot + self.config[DICE_GAMMA] # 분자
            denom = (prob ** 2) + (labels_onehot ** 2) + self.config[DICE_GAMMA] # 분모
            loss = 1 - (nom / denom)
            loss = 1000 * tf.reduce_mean(loss)
            loss = tf.reduce_mean(loss)
            loss *= 1000
            
            # loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prob)

            mask_bool = tf.cast(mask[:, :, 0], tf.bool)
            # pick only non padding values and flatten sequences
            labels_flat = tf.boolean_mask(labels, mask_bool)
            preds_flat = tf.boolean_mask(preds, mask_bool)
            # set `0` prediction to not a prediction
            labels_flat_one_hot = tf.one_hot(labels_flat - 1, self._num_tags - 1)
            preds_flat_one_hot = tf.one_hot(preds_flat - 1, self._num_tags - 1)
            
            losses.append(loss)
            entity_f1 = self._tf_layers['entity_f1'](labels_flat_one_hot, preds_flat_one_hot)
            self.entity_loss.update_state(loss)
            self.entity_f1.update_state(entity_f1)

        losses = tf.add_n(losses)
        return losses
    
    def batch_predict(self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, tf.Tensor]:
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.predict_data_signature)

        mask = tf_batch_data[TEXT_MASK][0]
        attn_mask = mask[:, None, None, :, 0]
        sequence_lengths = self._get_sequence_lengths(mask)
        
        inputs = self._tf_layers[f"input_layer"](tf_batch_data[TEXT_FEATURES], mask, sparse_dropout=self.config[SPARSE_INPUT_DROPOUT], training=self._training)
        x = self._tf_layers[f"base_layer"](inputs, 1 - attn_mask, self._training)
        
        out = {}
        if self.config[INTENT_CLASSIFICATION]:
            preds = self._tf_layers['intent_layer'](x, sequence_lengths, training=self._training)
            out['i_scores'] = preds
        
        if self.config[ENTITY_RECOGNITION]:
            _, preds = self._tf_layers['entity_layer'](x, sequence_lengths-1, training=self._training)
            out['e_ids'] = preds
        return out
# pytype: enable=key-error
