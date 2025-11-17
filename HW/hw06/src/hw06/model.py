from flax import nnx
import jax
import jax.numpy as jnp

class MultiHeadAttention(nnx.Module):
    """MultiHeadAttention class for self-attention, cross-attention and masked/unmasked"""
    def __init__(
            self,
            model_dim: int,
            head_size: int,
            rngs: nnx.Rngs,
    ):
        assert model_dim % head_size == 0, "model_dim should be divisible by head size"
        self.model_dim = model_dim
        self.head_size = head_size
        self.d_k = model_dim // head_size
        self.Wq = nnx.Linear(model_dim, model_dim, rngs=rngs)
        self.Wk = nnx.Linear(model_dim, model_dim, rngs=rngs)
        self.Wv = nnx.Linear(model_dim, model_dim, rngs=rngs)
        self.Wo = nnx.Linear(model_dim, model_dim, rngs=rngs)

    def attention(self, Q, K, V, mask: jnp.ndarray | None):
        attention = (Q @ jnp.swapaxes(K, -1, -2)) / jnp.sqrt(self.d_k)

        # handling case for masked attention
        if mask is not None:
            attention = jnp.where(mask, attention, attention - 1e9)
        attention = jax.nn.softmax(attention, axis=-1)
        output = attention @ V
        return output

    def split_head(self, x: jnp.ndarray):
        # splitting head for Multihead attention
        B, T, D = x.shape
        H, d_k = self.head_size, self.d_k
        x = x.reshape(B, T, H, d_k)       # (B, T, H, d_k)
        x = jnp.swapaxes(x, 1, 2)         # (B, H, T, d_k)
        return x
    
    def combine_head(self, x: jnp.ndarray):
        # combining heads after computing attentions
        B, H, T, d_k = x.shape
        x = jnp.swapaxes(x, 1, 2)         # (B, T, H, d_k)
        return x.reshape(B, T, H * d_k)
    
    def __call__(self, x: jnp.ndarray, *, cross: jnp.ndarray | None = None, mask: jnp.ndarray | None = None):
        Q = self.Wq(x)
        
        # Handling case for cross attention
        y = x if cross is None else cross
        K = self.Wk(y)
        V = self.Wv(y)

        Q = self.split_head(Q)
        K = self.split_head(K)
        V = self.split_head(V)
        attention_value = self.attention(Q, K, V, mask)

        output = self.Wo(self.combine_head(attention_value))
        return output

class FeedForward(nnx.Module):
    def __init__(
                self,
                n_embd: int,
                rngs: nnx.Rngs
    ):
        # From paper, input and output is d_model = 512 and d_ff = 4*d_model = 2048
        # Implemented the same way with paper
        self.linear1 = nnx.Linear(n_embd, 4 * n_embd, rngs=rngs)
        self.linear2 = nnx.Linear(4 * n_embd, n_embd, rngs=rngs)
    def __call__(self, x):
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x

class EncoderBlock(nnx.Module):
    def __init__(
            self,
            model_dim: int,
            head_size: int,
            *,
            rngs: nnx.Rngs
    ):
        self.head_size = head_size
        self.model_dim = model_dim
        self.attention = MultiHeadAttention(model_dim, head_size, rngs)
        self.ff = FeedForward(model_dim, rngs)
        self.norm1 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(model_dim, rngs=rngs)
    
    def __call__(self, x):
        # For encoder block, self attention and unmasked
        y = x + self.attention(x, cross=None, mask=None)
        x1 = self.norm1(y)
        z = x1 + self.ff(x1)
        x2 = self.norm2(z)
        return x2

class DecoderBlock(nnx.Module):
    def __init__(
            self,
            model_dim: int,
            head_size: int,
            *,
            rngs: nnx.Rngs
    ):
        self.head_size = head_size
        self.model_dim = model_dim
        self.attention = MultiHeadAttention(model_dim, head_size, rngs)
        self.ff = FeedForward(model_dim, rngs)
        self.norm1 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(model_dim, rngs=rngs)
        self.norm3 = nnx.LayerNorm(model_dim, rngs=rngs)

    def __call__(self, x, x_encoder, mask):
        # For decoder block, first attention is masked self-attention 
        y1 = x + self.attention(x, cross=None, mask=mask)
        x1 = self.norm1(y1)
        # second attention is unmasked cross attention
        y2 = x1 + self.attention(x1, cross=x_encoder, mask=None)
        x2 = self.norm2(y2)
        y3 = x2 + self.ff(x2)
        output = self.norm3(y3)
        return output

class Transformer(nnx.Module):
    def __init__(
            self,
            model_dim: int,
            head_size: int,
            vocab_size: int,
            num_layers: int,
            *,
            rngs: nnx.Rngs, 
    ):
        self.model_dim = model_dim
        self.head_size = head_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.x_embed = nnx.Embed(num_embeddings=vocab_size, features=model_dim, rngs=rngs)
        self.y_embed = nnx.Embed(num_embeddings=vocab_size, features=model_dim, rngs=rngs)

        self.EncoderBlocks = [EncoderBlock(model_dim, head_size, rngs) for _ in range(num_layers)]
        self.DecoderBlocks = [DecoderBlock(model_dim, head_size, rngs) for _ in range(num_layers)]
        self.Linear = nnx.Linear(model_dim, vocab_size, rngs=rngs)
    
    def PositionalEncoding(self, seq_length:int , model_dim: int):
        positions = jnp.arange(seq_length)[:, None]
        dims = jnp.arange(model_dim)[None,:] # if model_dim = 8, [0,1,2 ... ,8]

        # since embedding is sin for even and cos for odd, we need i = dims//2 to create
        # [0,0,1,1,2,2...]
        pos_enc = positions / (10000 ** ( (2 * dims//2) / model_dim))
        pos_enc = jnp.where(dims % 2 == 0, jnp.sin(pos_enc), jnp.cos(pos_enc))
        return pos_enc
    
    def __call__ (self, x: jnp.ndarray , y: jnp.ndarray):
        x = self.x_embed(x) * jnp.sqrt(self.model_dim)
        Bx, Tx, Dx = x.shape
        
        x_pos_enc = self.PositionalEncoding(Tx, self.model_dim)
        x = x + x_pos_enc[None, :, :]

        for block in self.EncoderBlocks:
            x = self.block(x)
        encoder_out = x

        y = self.y_embed(y) * jnp.sqrt(self.model_dim)
        By, Ty, Dy = y.shape

        y_pos_enc = self.PositionalEncoding(Ty, self.model_dim)
        y = y + y_pos_enc[None, :, :]
        self_mask = jnp.tril(jnp.ones((Ty, Ty), dtype=bool))[None, None, :, :]

        for block in self.DecoderBlocks:
            y = self.block(y, encoder_out, self_mask)
        decoder_out = y
        logits = self.Linear(decoder_out)
        return logits