# __init__.py
import structlog
from flax import nnx
import jax
import jax.numpy as jnp

from .logging import configure_logging
from .test import grad_future_leak_self_attn, check_perm_equivariance_mha
from .model import EncoderBlock, MultiHeadAttention

def main() -> None:
    """CLI entry point."""
    configure_logging()
    log = structlog.get_logger()

    # Creating data and MultiHeadAttention (MHA)
    B, T, D, H = 2, 6, 64, 8
    key = jax.random.key(0)
    key_model, key_x = jax.random.split(key)

    x = jax.random.normal(key_x, (B, T, D))
    perm = jnp.array([2, 0, 4, 1, 5, 3])
    mha = MultiHeadAttention(model_dim=D, head_size=H, rngs=nnx.Rngs(params=key_model))

    # Testing Permutation equivariance 
    log.info("perm_check.mha.start", B=B, T=T, D=D, H=H)
    passed_mha = check_perm_equivariance_mha(mha, x, perm)
    log.info("perm_check.mha", passed=bool(passed_mha))

    # Testing gradient for masked MHA
    key = jax.random.key(1)

    # testing gradient at t_mid
    t_mid = T // 2

    future_masked = grad_future_leak_self_attn(mha, T, D, t_mid, use_mask=True)
    future_unmasked = grad_future_leak_self_attn(mha, T, D, t_mid, use_mask=False)

    max_future_masked = float(jnp.max(jnp.abs(future_masked)))     # ~ 0 with mask
    max_future_unmasked = float(jnp.max(jnp.abs(future_unmasked)))   # > 0 without mask

    log.info("future grad matrices for masked MHA", masked=str(future_masked.tolist()))
    log.info("future grad matrices for unmasked MHA", unmasked=str(future_unmasked.tolist()))

    # Log the summary scalars
    log.info("max value of future grad matrices", max_masked=max_future_masked, max_unmasked=max_future_unmasked)

    # Masked case: expect = 0
    log.info("future_grad_check", case="masked", passed=(max_future_masked == 0), max_future=max_future_masked,)

    # Unmasked case: expect > 0
    log.info("future_grad_check", case="unmasked", passed=(max_future_unmasked > 0), max_future=max_future_unmasked)
