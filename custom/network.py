import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from rasa.utils.tensorflow import layers

from typing import Any, Dict, List, Optional, Text, Tuple, Union, Type

class InputLayer(tf.keras.layers.Layer):
    def __init__(self, dense_dim: List[int], model_dim: int, reg_lambda:float, drop_rate:float):
        super(InputLayer, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(i, activation='relu') for i in dense_dim]
        self.sparse_dropout_layer = layers.SparseDropout(drop_rate)
        self.sparse_to_dense_layer = layers.DenseForSparse(units=dense_dim[0], reg_lambda=reg_lambda)
        self.output_layer = tf.keras.layers.Dense(model_dim, activation='relu')

    def call(self, 
            features: List[Union[np.ndarray, tf.Tensor, tf.SparseTensor]], 
            mask: tf.Tensor, 
            sparse_dropout: bool = False, 
            training: Optional[Union[tf.Tensor, bool]] = None) -> tf.Tensor:

        dense_features = []
        for f in features:
            if isinstance(f, tf.SparseTensor):
                if sparse_dropout:
                    f = self.sparse_dropout_layer(f)
                f = self.sparse_to_dense_layer(f)
            dense_features.append(f)

        x = tf.concat(dense_features, axis=-1) * mask
        for d in self.dense_layers:
            x = d(x)
        x = self.output_layer(x)
        return x


class BaseLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, ffn_dim, num_head, drop_rate, num_layer):
        super(BaseLayer, self).__init__()
        self.trm = [TransformerLayer(model_dim, ffn_dim, num_head, drop_rate) for _ in range(num_layer)]
    
    def call(self, x: tf.Tensor, pad_mask: Optional[tf.Tensor] = None, training: Optional[Union[tf.Tensor, bool]] = None) -> tf.Tensor:
        for t in self.trm:
            x = t(x, mask=pad_mask, training=training)
        return x


class IntentLayer(tf.keras.layers.Layer):
    def __init__(self, num_intent):
        super(IntentLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(num_intent, activation='softmax')

    def call(self, x: tf.Tensor, sequence_lengths: tf.Tensor, training: Optional[Union[tf.Tensor, bool]] = None) -> tf.Tensor:
        last_token_index = tf.maximum(0, sequence_lengths - 1)
        batch_index = tf.range(tf.shape(last_token_index)[0])
        indices = tf.stack([batch_index, last_token_index], axis=1)
        x = tf.gather_nd(x, indices)
        x = self.dense(x)
        return x
    
class EntityLayer(tf.keras.layers.Layer):
    def __init__(self, num_tag):
        super(EntityLayer, self).__init__()
        self.num_tag = num_tag
        self.dense = tf.keras.layers.Dense(num_tag)
    
    def build(self, input_shape: tf.TensorShape) -> None:
        self.transition_params = self.add_weight(shape=(self.num_tag, self.num_tag), name='transition params')
        self.built = True

    def call(self, x: tf.Tensor, y: tf.Tensor, sequence_lengths: tf.Tensor, training: Optional[Union[tf.Tensor, bool]] = None):
        x = self.dense(x)
        
        if training:
            log_likelihood, _ = tfa.text.crf_log_likelihood(x, y, sequence_lengths, self.transition_params)
        else:
            log_likelihood = None

        preds, _ = tfa.text.crf.crf_decode(x, self.transition_params, sequence_lengths)
        mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(preds)[1], dtype=preds.dtype)
        preds = preds * mask

        return log_likelihood, preds


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_head):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        self.projection_dim = self.model_dim // self.num_head
        assert self.model_dim % self.num_head == 0

        self.qw = tf.keras.layers.Dense(self.model_dim)
        self.kw = tf.keras.layers.Dense(self.model_dim)
        self.vw = tf.keras.layers.Dense(self.model_dim)
        self.w = tf.keras.layers.Dense(self.model_dim)
    
    def attention(self, q, k ,v, mask):
        dim = tf.cast(q.shape[-1], tf.float32)
        score = tf.matmul(q, k, transpose_b=True)
        scaled_score = score / tf.math.sqrt(dim)

        if mask is not None:
            scaled_score += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_score)
        attention_outputs = tf.matmul(attention_weights, v)
        return attention_outputs, attention_weights
    
    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_head, self.projection_dim))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x
    
    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, -1, self.model_dim))
        return x
    
    def call(self, q, k, v, mask):
        q, k, v = self.qw(q), self.kw(k), self.vw(v)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        outputs, weights = self.attention(q, k, v, mask)
        outputs = self.combine_heads(outputs)
        outputs = self.w(outputs)
        return outputs

class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, model_dim, ffn_dim):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(ffn_dim)
        self.dense2 = tf.keras.layers.Dense(model_dim)

    def call(self, x):
        x = self.dense1(x)
        x = tfa.activations.gelu(x)
        x = self.dense2(x)
        return x

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, ffn_dim, num_head, drop_rate):
        super(TransformerLayer, self).__init__()
        self.mha = MultiHeadAttention(model_dim, num_head)
        self.ffn = FeedForwardNetwork(model_dim, ffn_dim)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        out1 = self.mha(x, x, x, mask)
        out1 = self.dropout1(out1, training=training)
        out1 = self.layernorm1(x + out1)
        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training=training)
        out2 = self.layernorm2(out1 + out2)
        return out2

class FlairEmbedding(tf.keras.Model):
    def __init__(self, vocab_size, model_dim):
        super(FlairEmbedding, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_dim)
        self.forward_lstm = tf.keras.layers.LSTM(model_dim, return_sequences=True)
        self.forward_outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')
        self.backward_lstm = tf.keras.layers.LSTM(model_dim, return_sequences=True, go_backwards=True)
        self.backward_outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')
    
    def call(self, x, training):
        x = self.embedding(x)
        forward = self.forward_lstm(x)
        forward_outputs = self.forward_outputs(forward)
        backward = self.backward_lstm(x)
        backward_outputs = self.backward_outputs(backward)
        return forward_outputs, backward_outputs
    
    def train_step(self, data):
        x, (forward_y, backward_y) = data

        with tf.GradientTape() as tape:
            forward_outputs, backward_outputs = self(x, training=True)  # Forward pass
            forward_loss = self.compiled_loss(forward_y, forward_outputs)
            backward_loss = self.compiled_loss(backward_y, backward_outputs)
            loss = forward_loss + backward_loss
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return

    def get_representation(self, x):
        x = self.embedding(x)
        forward = self.forward_lstm(x)
        backward = self.backward_lstm(x)
        return forward, backward