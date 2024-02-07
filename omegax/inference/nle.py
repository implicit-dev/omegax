import jax
import jax.numpy as jnp
import tensorflow_probability as tfp

from omegax.inference.nde.utils import ConditionnalNDE

tfp = tfp.experimental.substrates.jax


class NLE(ConditionnalNDE):
    def __init__(self):
        self.method = "NLE"

    def sample(
        self,
        log_prob_prior,
        observation,
        init_point,
        key,
        num_results=3e4,
        num_burnin_steps=5e2,
        num_chains=12,
    ):
        print("... running hmc")

        @jax.vmap
        def unnormalized_log_prob(theta):
            prior = log_prob_prior(theta)

            likelihood = self.log_prob_fn(
                self.params,
                theta.reshape([1, self.dim]),
                jnp.array(observation).reshape([1, self.dim]),
            )

            return likelihood + prior

        # Initialize the HMC transition kernel.
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_log_prob,
                num_leapfrog_steps=3,
                step_size=1e-2,
            ),
            num_adaptation_steps=int(num_burnin_steps * 0.8),
        )

        # Run the chain (with burn-in).
        # @jax.jit
        def run_chain():
            # Run the chain (with burn-in).
            samples, is_accepted = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=jnp.array(init_point) * jnp.ones([num_chains, self.dim]),
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
                seed=key,
            )

            return samples, is_accepted

        samples_hmc, is_accepted_hmc = run_chain()
        sample_nd = samples_hmc[is_accepted_hmc]

        print("done ✓")

        return sample_nd
