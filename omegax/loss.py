import jax
import jax.numpy as jnp
import tensorflow_probability as tfp

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions


def mse(y, theta, dim=None, log_prob_fn=None):
    """Compute the Mean Squared Error loss"""
    return jnp.mean(jnp.sum((y - theta) ** 2, axis=1))


def mae(y, theta, dim=None, log_prob_fn=None):
    """Compute the Mean Absolute Error loss"""
    return jnp.mean(jnp.sum(jnp.absolute(y - theta), axis=1))


def nll(y, theta, log_prob_fn, dim=None):
    """Compute the negative log likelihood loss"""
    return -jnp.mean(log_prob_fn(theta, y))


def gnll(y, theta, dim, log_prob_fn=None):
    """Compute the Gaussian Negative Log Likelihood loss"""
    y_mean = y[..., :dim]
    y_var = y[..., dim:]
    y_var = tfb.FillScaleTriL(diag_bijector=tfb.Softplus(low=1e-3)).forward(y_var)

    @jax.jit
    @jax.vmap
    def _get_log_prob(y_mean, y_var, theta):
        likelihood = tfd.MultivariateNormalTriL(y_mean, y_var)
        return likelihood.log_prob(theta)

    loss = -jnp.mean(_get_log_prob(y_mean, y_var, theta))

    return loss
