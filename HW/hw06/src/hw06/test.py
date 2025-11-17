import jax
import jax.numpy as jnp
from flax import nnx


def check_perm_equivariance_mha(mha, x, perm):
    # No mask, no PE
    y = mha(x)                        # (B,T,D)
    y_perm_in = mha(x[:, perm, :])    # permute input
    y_perm_out = y[:, perm, :]        # permute outputs
    return jnp.allclose(y_perm_in, y_perm_out, atol=1e-5)

def causal_mask(T: int) -> jnp.ndarray:
    return jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]

def grad_future_leak_self_attn(mha, Ty: int, D: int, t: int, use_mask: bool):
    key = jax.random.key(0)
    y_in = jax.random.normal(key, (1, Ty, D))
    mask = causal_mask(Ty) if use_mask else None

    def f(y_):
        y_out = mha(y_, mask=mask)          # self-attn: Q,K,V from y_
        return y_out[:, t, :].sum()

    g = jax.grad(lambda y: f(y).sum())(y_in)  # (1, Ty, D)
    future = g[:, (t+1):, :]                  # grads wrt future positions
    return future

