import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability as tfp
from tqdm import tqdm

from omegax.loss import nll

tfp = tfp.experimental.substrates.jax


class ConditionnalNDE:
    def __init__(self, NDE, dim, method):
        self.NDE = NDE
        self.dim = dim
        self.method = method
        self.log_prob_fn = self.build_nde()

    def build_nde(self):
        if self.method == "NPE":
            return hk.without_apply_rng(
                hk.transform(
                    lambda theta, y: self.NDE(self.dim)(y).log_prob(theta).squeeze()
                )
            )
        elif self.method == "NLE":
            return hk.without_apply_rng(
                hk.transform(
                    lambda theta, y: self.NDE(self.dim)(theta).log_prob(y).squeeze()
                )
            )

    def initialized_params(self, theta, y, key):
        self.params = self.NDE.init(
            key, theta * jnp.ones([1, self.dim]), y * jnp.ones([1, self.dim])
        )

    def loss(self, params, theta, y):
        def log_prob(theta, y):
            return self.log_prob_fn.apply(params, theta, y)

        return nll(y=y, theta=theta, dim=self.dim, log_prob_fn=log_prob)

    def train(self, data, learning_rate, total_steps=30_000, batch_size=128):
        dataset_theta = data["theta"]
        dataset_y = data["y"]

        nb_simu = len(dataset_theta)

        print("nb of simulations used for training: ", nb_simu)

        params = self.params
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        @jax.jit
        def update(params, opt_state, theta, y):
            """Single SGD update step."""
            loss, grads = jax.value_and_grad(self.loss)(params, theta, y)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return loss, new_params, new_opt_state

        print("... start training")

        batch_loss = []
        pbar = tqdm(range(total_steps))

        for batch in pbar:
            inds = np.random.randint(0, nb_simu, batch_size)
            ex_theta = dataset_theta[inds]
            ex_y = dataset_y[inds]

            if not jnp.isnan(ex_y).any():
                l, params, opt_state = update(params, opt_state, ex_theta, ex_y)

                batch_loss.append(l)
                pbar.set_description(f"loss {l:.3f}")

                if jnp.isnan(l):
                    break

        self.params = params
        self.loss = batch_loss

        print("done ✓")
