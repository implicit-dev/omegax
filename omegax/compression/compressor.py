from functools import partial

import jax
import optax
import tensorflow_probability as tfp

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions


class Compressor:
    def __init__(
        self,
        compressor,
        nf,
        optimizer,
        loss_fn,
        dim=None,
    ):
        self.compressor = compressor
        self.nf = nf
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dim = dim

    def loss(self, params, theta, x, state_resnet):
        y, opt_state_resnet = self.compressor.apply(params, state_resnet, None, x)

        if self.nf is not None:

            def log_prob_fn(theta, y):
                return self.nf.apply(params, theta, y)

        loss = self.loss_fn(y=y, theta=theta, dim=self.dim, log_prob_fn=log_prob_fn)
        return loss, opt_state_resnet

    @partial(jax.jit, static_argnums=(0,))
    def update(self, model_params, opt_state, theta, x, state_resnet=None):
        (loss, opt_state_resnet), grads = jax.value_and_grad(self.loss, has_aux=True)(
            model_params, theta, x, state_resnet
        )

        updates, new_opt_state = self.optimizer.update(grads, opt_state)

        new_params = optax.apply_updates(model_params, updates)

        return loss, new_params, new_opt_state, opt_state_resnet
