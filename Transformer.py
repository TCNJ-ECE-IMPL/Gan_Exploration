#Transformer

image = '/home/nallurn1/VQGAN/images/0_2000.jpg'

patches = tf.image.extract_patches(
            images=images[:1]25_2000.jpg25_2000.jpg,
            sizes=[1, 8, 8, 1],
            strides=[1, 8, 8, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Transformer block multi-head Self Attention
        self.multiheadselfattention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
def call(self, inputs, training):
        out1 = self.layernorm1(inputs)       
        attention_output = self.multiheadselfattention(out1)
        attention_output = self.dropout1(attention_output, training=training)       
        out2 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out2 + ffn_output)