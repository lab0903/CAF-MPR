import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm_a = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_b = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_a = layers.Dropout(rate)
        self.dropout_b = layers.Dropout(rate)

    def call(self, inputs, training=True):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout_a(attn_output, training=training)
        out_a = self.layernorm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output, training=training)
        return self.layernorm_b(out_a + ffn_output)


class MultimodalAttention7(layers.Layer):
    def __init__(self, embed_dim, num_heads, rate=0.1):
        super(MultimodalAttention7, self).__init__()
        # 定义三个模态之间的交叉注意力层
        self.trace_to_others = self._build_attention(num_heads, embed_dim)
        self.er_to_others = self._build_attention(num_heads, embed_dim)
        self.pm_to_others = self._build_attention(num_heads, embed_dim)

        # 正则化和融合组件
        self.dropout = layers.Dropout(rate)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dense_fusion = layers.Dense(embed_dim)

    def _build_attention(self, num_heads, embed_dim):
        """创建针对两模态的交叉注意力模块"""
        return layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            attention_axes=(1,)  # 注意序列轴（如时间或空间）
        )

    def _create_attention_mask(self, embed, pad_token):
        mask = tf.cast(embed != pad_token, tf.float32)  # 1表示有效部分，0表示填充部分
        return mask

    def _cross_attention_block(self, query, source1, source2, attention_layer, training=True):
        """两源交叉注意力融合"""
        # Source1作为Key/Value，提供信息
        attn1 = attention_layer(query=query, key=source1, value=source1)
        attn1 = self.dropout(attn1, training=training)
        # Source2作为附加Key/Value（可选）
        attn2 = attention_layer(query=query, key=source2, value=source2)
        attn2 = self.dropout(attn2, training=training)
        # 合并两路结果
        merged = (attn1 + attn2) / 2  # 或Concat后进行线性层
        # merged = self.dense_fusion(merged) #新增
        return self.layernorm(query + merged)

    def call(self, trace_embed, second_embed, training=True):
        trace_enhanced = self._cross_attention_block(
            trace_embed, second_embed, second_embed, self.trace_to_others, training
        )

        second_enhanced = self._cross_attention_block(
            second_embed, trace_embed, trace_embed, self.er_to_others, training
        )

        combined = tf.concat([trace_enhanced, second_enhanced], axis=-1)
        output = self.dense_fusion(combined)  # 维度调整
        return output


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        # Get the sequence length
        seq_len = tf.shape(x)[-1]
        # Token Embedding and Position Embedding
        position_indices = tf.range(start=0, limit=seq_len, delta=1)
        pos_embed = self.pos_emb(position_indices)
        token_embed = self.token_emb(x)
        print(pos_embed.shape)
        print(token_embed.shape)

        return token_embed + pos_embed


class TokenAndPositionEmbedding2(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding2, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.maxlen = maxlen
        self.nodes = vocab_size - 2  # pad & unk

    def call(self, x, second_embed=None):
        # Get the sequence length
        seq_len = tf.shape(x)[-1]

        # Token Embedding and Position Embedding
        token_embed = self.token_emb(x)
        position_indices = tf.range(start=0, limit=seq_len, delta=1)
        pos_embed = self.pos_emb(position_indices)

        # Normalize token embedding and position embedding
        # token_embed = self.norm(token_embed)
        # pos_embed = self.norm(pos_embed)

        token_pos = token_embed + pos_embed

        # If relation_embed is provided, reshape to match token_embed shape
        if second_embed is not None:
            # Expand relation_embed to match the sequence length
            second_embed = tf.expand_dims(second_embed, axis=0)
            second_embed = tf.tile(second_embed, [tf.shape(x)[0], 1, 1])  # [batch_size, seq_len, embed_dim]

            if self.maxlen - self.nodes > 0:
                second_embed = tf.pad(second_embed, [[0, 0], [0, self.maxlen - self.nodes], [0, 0]])
            elif self.maxlen - self.nodes < 0:
                token_pos = tf.pad(token_pos, [[0, 0], [0, self.nodes - self.maxlen], [0, 0]])

        return token_pos, second_embed


class TokenAndPositionEmbedding3(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding3, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.maxlen = maxlen
        self.nodes = vocab_size - 2  # pad & unk

    def call(self, x, er_embed=None, pm_embed=None):
        # Get the sequence length
        seq_len = tf.shape(x)[-1]

        # Token Embedding and Position Embedding
        token_embed = self.token_emb(x)
        position_indices = tf.range(start=0, limit=seq_len, delta=1)
        pos_embed = self.pos_emb(position_indices)

        # Add batch dimension to pos_embed
        pos_embed = tf.expand_dims(pos_embed, axis=0)  # [1, seq_len, embed_dim]
        pos_embed = tf.tile(pos_embed, [tf.shape(x)[0], 1, 1])  # [batch_size, seq_len, embed_dim]

        # Normalize token embedding and position embedding
        token_embed = self.norm(token_embed)
        pos_embed = self.norm(pos_embed)
        #token_pos = token_embed + pos_embed
        token_pos = token_embed

        # If relation_embed is provided, reshape to match token_embed shape
        if er_embed is not None:
            # Expand relation_embed to match the sequence length
            er_embed = tf.expand_dims(er_embed, axis=0)
            er_embed = tf.tile(er_embed, [tf.shape(x)[0], 1, 1])  # [batch_size, seq_len, embed_dim]
            if self.maxlen - self.nodes > 0:
                er_embed = tf.pad(er_embed, [[0, 0], [0, self.maxlen - self.nodes], [0, 0]])
            elif self.maxlen - self.nodes < 0:
                token_pos = tf.pad(token_pos, [[0, 0], [0, self.nodes - self.maxlen], [0, 0]])

        er_embed = self.norm(er_embed)

        if pm_embed is not None:
            pm_embed = tf.expand_dims(pm_embed, axis=0)
            pm_embed = tf.tile(pm_embed, [tf.shape(x)[0], 1, 1])
            if self.maxlen - self.nodes > 0:
                pm_embed = tf.pad(pm_embed, [[0, 0], [0, self.maxlen - self.nodes], [0, 0]])

        pm_embed = self.norm(pm_embed)

        return token_pos, er_embed, pm_embed


# 6，本文方法
class MultimodalAttention6(layers.Layer):
    def __init__(self, embed_dim, num_heads, vocab_size, rate=0.1):
        super(MultimodalAttention6, self).__init__()
        # 定义三个模态之间的交叉注意力层
        self.trace_to_others = self._build_attention(num_heads, embed_dim)
        self.er_to_others = self._build_attention(num_heads, embed_dim)
        self.pm_to_others = self._build_attention(num_heads, embed_dim)

        # 正则化和融合组件
        self.dropout = layers.Dropout(rate)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dense_fusion = layers.Dense(embed_dim)

    def _build_attention(self, num_heads, embed_dim):
        """创建针对两模态的交叉注意力模块"""
        return layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            attention_axes=(1,)  # 注意序列轴（如时间或空间）
        )

    def _cross_attention_block(self, query, source1, source2, attention_layer, training=True):
        """两源交叉注意力融合"""
        # Source1作为Key/Value，提供信息
        attn1 = attention_layer(query=query, key=source1, value=source1)
        attn1 = self.dropout(attn1, training=training)
        # Source2作为附加Key/Value（可选）
        attn2 = attention_layer(query=query, key=source2, value=source2)
        attn2 = self.dropout(attn2, training=training)
        # 合并两路结果
        merged = (attn1 + attn2) / 2  # 或Concat后进行线性层

        return self.layernorm(query + merged)

    def call(self, trace_embed, er_embed, pm_embed, training=True):
        # Trace模态为Query
        trace_enhanced = self._cross_attention_block(
            trace_embed, er_embed, pm_embed, self.trace_to_others, training
        )

        # er模态为Query
        er_enhanced = self._cross_attention_block(
            er_embed, trace_embed, pm_embed, self.er_to_others, training
        )

        # pm模态为Query
        pm_enhanced = self._cross_attention_block(
            pm_embed, trace_embed, er_embed, self.pm_to_others, training
        )

        combined = tf.concat([trace_enhanced, er_enhanced, pm_enhanced], axis=-1)
        output = self.dense_fusion(combined)  # 维度调整

        print("***Full Model***")
        return output


def get_next_activity_model(max_case_length, vocab_size, output_dim,
                            er_embed, pm_embed, embed_dim=36, num_heads=2):
    # Input layer
    inputs = layers.Input(shape=(max_case_length,))

    # Multimodal Attention to combine Transformer embedding and RGCN embedding
    if er_embed is not None and pm_embed is not None:
        token_position_embedding, event_relation_embedding, process_model_embdding = TokenAndPositionEmbedding3(
            max_case_length, vocab_size, embed_dim)(inputs, er_embed, pm_embed)
        ma_output = MultimodalAttention6(embed_dim, num_heads, vocab_size)(token_position_embedding, event_relation_embedding, process_model_embdding)
    elif er_embed is not None and pm_embed is None:
        token_position_embedding, event_relation_embedding = TokenAndPositionEmbedding2(
            max_case_length, vocab_size, embed_dim)(inputs, er_embed)
        token_position_embedding = layers.LayerNormalization(epsilon=1e-6)(token_position_embedding)
        ma_output = MultimodalAttention7(embed_dim, num_heads)(token_position_embedding, event_relation_embedding)
    elif er_embed is None and pm_embed is not None:
        token_position_embedding, process_model_embdding = TokenAndPositionEmbedding2(
            max_case_length, vocab_size, embed_dim)(inputs, pm_embed)
        token_position_embedding = layers.LayerNormalization(epsilon=1e-6)(token_position_embedding)
        ma_output = MultimodalAttention7(embed_dim, num_heads)(token_position_embedding, process_model_embdding)
    else:
        ma_output = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)

    x = layers.Dense(embed_dim, activation="relu")(ma_output)

    if er_embed is None and pm_embed is None:
        print("---only TRANSFORMER---")
        x = TransformerBlock(embed_dim, num_heads, 64)(x)

    # Pooling, dropout, and output layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim)(x)

    # Construct the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="next_activity_multi_attention")
    return model