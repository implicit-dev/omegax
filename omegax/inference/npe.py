import haiku as hk
import jax.numpy as jnp
import tensorflow_probability as tfp

from omegax.inference.nde.utils import ConditionnalNDE

tfp = tfp.experimental.substrates.jax


class NPE(ConditionnalNDE):
    def __init__(self):
        self.method = "NPE"

    def sample(self, observation, nb_samples, key):
        hk.transform(
            lambda y: self.NDE(self.dim)(y).sample(nb_samples, seed=hk.next_rng_key())
        )

        sample_nd = self.nde_sample_fn.apply(
            self.params, rng=key, y=observation * jnp.ones([nb_samples, self.dim])
        )

        return sample_nd
