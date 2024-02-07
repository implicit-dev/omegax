import haiku as hk
import jax.numpy as jnp
import tensorflow_probability as tfp
from bijectors import AffineCoupling

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions


class ConditionalRealNVP(hk.Module):
    """A normalizing flow based on RealNVP using specified bijector functions."""

    def __init__(self, d, *args, n_layers=3, bijector_fn=AffineCoupling, **kwargs):
        """
        Args:
        d, dimensionality of the input
        n_layers, number of layers
        coupling_layer, list of coupling layers
        """
        self.d = d
        self.n_layer = n_layers
        self.bijector_fn = bijector_fn
        super().__init__(*args, **kwargs)

    def __call__(self, y):
        chain = tfb.Chain(
            [
                tfb.Permute(jnp.arange(self.d)[::-1])(
                    tfb.RealNVP(
                        self.d // 2, bijector_fn=self.bijector_fn(y, name="b%d" % i)
                    )
                )
                for i in range(self.n_layer)
            ]
        )

        nvp = tfd.TransformedDistribution(
            tfd.MultivariateNormalDiag(0.5 * jnp.ones(self.d), 0.05 * jnp.ones(self.d)),
            bijector=chain,
        )

        return nvp
