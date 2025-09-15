import jax
import jax.numpy as jnp
from flax import nnx


class NNXLinearModel(nnx.Module):
    """Flax Linear Regression Model"""

    def __init__(self, *, rngs: nnx.Rngs, num_input: int, num_output: int):
        self.num_input = num_input
        self.num_output = num_output
        key = rngs.params()
        stddev = jnp.sqrt(2.0 / self.num_input)
        self.weights = nnx.Param(
            jax.random.normal(key, (self.num_input, num_output)) * stddev
        )
        self.bias = nnx.Param(jnp.zeros((1, num_output)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output for a given input."""
        return x @ self.weights.value + self.bias.value


class NNXMLP(nnx.Module):
    """A Flax NNX module for a MLP model"""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        num_input: int,
        num_output: int,
        hidden_layer_width: int,
        num_hidden_layers: int,
        hidden_activation=nnx.relu,
        output_activation=nnx.identity,
    ):
        @nnx.split_rngs(splits=num_hidden_layers + 2, only="params")
        def _split(rngs: nnx.Rngs):
            @nnx.vmap(in_axes=0, out_axes=0)
            def _one(r: nnx.Rngs):
                return r.params()

            return _one(rngs)

        keys = _split(rngs)
        self.num_input = num_input
        self.num_output = num_output
        self.hidden_layer_width = hidden_layer_width
        self.num_hidden_layers = num_hidden_layers

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.input_layer = NNXLinearModel(
            rngs=nnx.Rngs(keys[0]), num_input=num_input, num_output=hidden_layer_width
        )

        @nnx.vmap(in_axes=0, out_axes=0)
        def make_hidden(key):
            return NNXLinearModel(
                rngs=nnx.Rngs(params=key),
                num_input=hidden_layer_width,
                num_output=hidden_layer_width,
            )

        hidden_models = make_hidden(keys[1:-1])

        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        def apply_hidden(layer: NNXLinearModel, x):
            return self.hidden_activation(layer(x))

        self.hidden_layers = hidden_models
        self.apply_hidden = apply_hidden

        self.output_layer = NNXLinearModel(
            rngs=nnx.Rngs(keys[-1]), num_input=hidden_layer_width, num_output=num_output
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.hidden_activation(self.input_layer(x))
        x = self.apply_hidden(self.hidden_layers, x)
        x = self.output_activation(self.output_layer(x))
        return x
