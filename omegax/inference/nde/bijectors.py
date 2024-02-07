import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors


class AffineCoupling(hk.Module):
    def __init__(
        self, y, *args, layers=[128, 128], activation=jax.nn.leaky_relu, **kwargs
    ):
        """
        Args:
        y, conditioning variable
        layers, list of hidden layers
        activation, activation function for hidden layers
        """
        self.y = y
        self.layers = layers
        self.activation = activation
        super().__init__(*args, **kwargs)

    def __call__(self, x, output_units, **condition_kwargs):
        net = jnp.concatenate([x, self.y], axis=-1)
        for i, layer_size in enumerate(self.layers):
            net = self.activation(hk.Linear(layer_size, name="layer%d" % i)(net))

        shifter = tfb.Shift(hk.Linear(output_units)(net))
        scaler = tfb.Scale(jnp.clip(jnp.exp(hk.Linear(output_units)(net)), 1e-2, 1e2))
        return tfb.Chain([shifter, scaler])
