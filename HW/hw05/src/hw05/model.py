from flax import nnx
import jax
import jax.numpy as jnp

class Conv1DTextClassifier(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 4,
        embed_dim: int = 128,
        conv_features: int = 64,
        kernel_size: int = 5,
        dropout_rate: float = 0.2,
        rngs: nnx.Rngs = None,
    ):
        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=embed_dim,
            rngs=rngs,
        )
        self.conv1 = nnx.Conv(
            in_features=embed_dim,        
            out_features=conv_features,
            kernel_size=(kernel_size,),
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.linear = nnx.Linear(conv_features, num_classes, rngs=rngs)

    def __call__(self, input_ids, *, train=True):
        x = self.embedding(input_ids)         
        x = jax.nn.relu(self.conv1(x))        
        x = jnp.max(x, axis=1)                
        x = self.dropout(x, deterministic=bool(not train))
        logits = self.linear(x)               
        return logits
