import time
import numpy as np
from utils.dp_solver import solve_cdp, solve_ldp, solve_pairwise, solve_mixing


class DPNoiseGenerator:
    def __init__(
            self,
            dp_mechanism,
            epsilon,
            delta,
            norm_clip,
            n_clients,
            n_rounds,
            model_dim,
            weight_matrix=None,
            seed=None
    ):
        """
        Generates differential privacy noise based on the selected DP mechanism.

        Parameters:
        - dp_mechanism: str, differential privacy mechanism ('cdp', 'ldp', 'pairwise', 'mixing')
        - epsilon: float, privacy budget
        - delta: float, privacy parameter
        - norm_clip: float, gradient clip constant
        - n_rounds: int
        - n_rounds: int
        - model_dim: int
        - weight_matrix: np.ndarray, weight matrix (n_clients x n_clients)
        - seed: int

        Attributes:
        - noise_params:
            - For 'cdp': variance (float).
            - For other mechanisms: covariance matrix (n_clients x n_clients, np.ndarray).
        - noise: np.ndarray
            - For 'cdp': shape (n_rounds, n_params).
            - For other DP mechanisms: shape (n_rounds, n_params, n_clients).
        """
        self.dp_mechanism = dp_mechanism
        self.n_rounds = n_rounds
        self.noise_params = None
        self.noise = None

        # Seed the random generator
        rng_seed = seed if seed is not None else int(time.time())
        self.np_rng = np.random.default_rng(rng_seed)

        # Compute noise parameters
        self._init_noise_params(
            epsilon=epsilon,
            delta=delta,
            norm_clip=norm_clip,
            n_clients=n_clients,
            n_rounds=n_rounds,
            weight_matrix=weight_matrix
        )

        # Generate differential privacy noise
        self._generate_dp_noise(
            n_clients=n_clients,
            n_rounds=n_rounds,
            model_dim=model_dim
        )

    @staticmethod
    def _compute_dp_bound(epsilon, delta, norm_clip, n_rounds):
        """
        Compute the differential privacy bound. From: https://arxiv.org/abs/2405.01031
        Returns:
        - float, privacy bound.
        """
        log_delta = np.log(1 / delta)
        numerator = (np.sqrt(log_delta + epsilon) - np.sqrt(log_delta)) ** 2
        denominator = 2 * n_rounds * norm_clip ** 2

        return numerator / denominator

    def _init_noise_params(self, epsilon, delta, norm_clip, n_clients, n_rounds, weight_matrix):
        """
        Initialize noise parameters based on the DP mechanism.

        Sets:
        - self.noise_params:
            - For 'cdp': float, variance.
            - For other DP mechanisms: np.ndarray, covariance matrix of shape (n_clients, n_clients).
        """
        privacy_bound = self._compute_dp_bound(epsilon=epsilon, delta=delta, norm_clip=norm_clip, n_rounds=n_rounds)

        if self.dp_mechanism == 'cdp':
            self.noise_params = solve_cdp(n_clients, privacy_bound)
        elif self.dp_mechanism == 'ldp':
            self.noise_params = solve_ldp(n_clients, privacy_bound)
        elif self.dp_mechanism == 'pairwise':
            self.noise_params = solve_pairwise(weight_matrix, privacy_bound)
        elif self.dp_mechanism == 'mixing':
            self.noise_params = solve_mixing(weight_matrix, privacy_bound)
        else:
            raise ValueError(f"Unsupported DP mechanism: {self.dp_mechanism}")

        if self.noise_params is None:
            raise ValueError("Noise parameter computation failed.")

    def _generate_dp_noise(self, n_clients, n_rounds, model_dim):
        """
        Generates the noise for all rounds.

        Sets:
        - self.noise:
            - For 'cdp': shape (n_rounds, n_params).
            - For other DP mechanisms: shape (n_rounds, n_params, n_clients).
        """
        if self.dp_mechanism == 'cdp':
            self.noise = self.np_rng.normal(
                0, np.sqrt(self.noise_params), size=(n_rounds + 1, model_dim)
            )
        else:
            self.noise = self.np_rng.multivariate_normal(
                mean=np.zeros(n_clients),
                cov=self.noise_params,
                size=(n_rounds + 1, model_dim)
            )

    def get_dp_noise(self, c_round):
        """
        Get the differential privacy noise for a specific round.

        Parameters:
        - c_round: int

        Returns:
        - Numpy array:
            - For 'cdp': shape (n_params).
            - For other mechanisms: shape (n_params, n_clients).
        """
        if self.noise is None:
            raise ValueError("Noise has not been generated.")
        if c_round >= self.n_rounds + 1:
            raise IndexError("Iteration index out of range.")
        return self.noise[c_round]
