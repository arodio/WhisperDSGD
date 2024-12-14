import numpy as np
import cvxpy as cp
from utils.graph import get_laplacian

import warnings


def solve_cdp(n_clients, privacy_bound):
    """
    Get the variance for Central Differential Privacy (CDP).
    Parameters:
    - n_clients:
    - privacy_bound
    Returns:
    - sigma_cdp: float
    """
    if privacy_bound > 0:
        sigma_ldp = 1 / privacy_bound
        sigma_cdp = sigma_ldp / n_clients
        return sigma_cdp
    else:
        raise ValueError("Invalid privacy bound!!")


def solve_ldp(n_clients, privacy_bound):
    """
    Get the covariance matrix for Local Differential Privacy (LDP).
    Parameters:
    - n_clients:
    - privacy_bound:
    Returns:
    - covariance_matrix: numpy.ndarray
    """
    identity_matrix = np.eye(n_clients)

    # Compute sigma_ldp
    if privacy_bound > 0:
        sigma_ldp = 1 / privacy_bound
    else:
        raise ValueError("Invalid privacy bound!!")

    return sigma_ldp * identity_matrix


def solve_pairwise(weight_matrix, privacy_bound):
    """
    Solve the optimization problem for Pairwise Correlated Differential Privacy.
    Parameters:
    - mixing_matrix: (n x n) weight matrix
    - privacy_bound:
    Returns:
    - covariance_matrix: numpy.ndarray
    """
    n_clients = weight_matrix.shape[0]
    identity_matrix = np.eye(n_clients)
    laplacian_matrix = get_laplacian(weight_matrix)
    sigma_cdp = cp.Variable(nonneg=True)
    sigma_cor = cp.Variable(nonneg=True)

    # Define the covariance matrix
    covariance_matrix = sigma_cdp * identity_matrix + sigma_cor * laplacian_matrix

    # Define the optimization problem
    objective = cp.Minimize(cp.trace(weight_matrix @ covariance_matrix @ weight_matrix.T))
    constraints = [covariance_matrix >> 0] + [
        cp.bmat([[covariance_matrix, unit_vector[:, None]], [unit_vector[None, :], np.array([[privacy_bound]])]]) >> 0
        for unit_vector in identity_matrix
    ]

    # Solve the optimization problem
    results_dict = _solve_problem(objective, constraints)
    if results_dict is None:
        return None

    # Return optimal value
    sigma_cdp_opt = results_dict[sigma_cdp.name()]
    sigma_cor_opt = results_dict[sigma_cor.name()]
    covariance_matrix_opt = sigma_cdp_opt * identity_matrix + sigma_cor_opt * laplacian_matrix
    return covariance_matrix_opt


def solve_mixing(weight_matrix, privacy_bound):
    """
    Solve the optimization problem for Mixing-based Correlation Differential Privacy.
    Parameters:
    - weight_matrix: (n x n) weight matrix
    - privacy_bound:
    Returns:
    - covariance_matrix: numpy.ndarray
    """
    n_clients = weight_matrix.shape[0]
    identity_matrix = np.eye(n_clients)
    covariance_cor = cp.Variable((n_clients, n_clients), PSD=True)
    sigma_cdp = cp.Variable(nonneg=True)

    # Define the covariance matrix
    covariance_matrix = sigma_cdp * identity_matrix + covariance_cor

    # Define the optimization problem
    objective = cp.Minimize(cp.trace(weight_matrix @ covariance_matrix @ weight_matrix.T))
    constraints = [
        cp.bmat([[covariance_matrix, unit_vector[:, None]], [unit_vector[None, :], np.array([[privacy_bound]])]]) >> 0
        for unit_vector in identity_matrix
    ]

    # Solve the optimization problem
    results_dict = _solve_problem(objective, constraints)
    if results_dict is None:
        return None

    # Return optimal values
    sigma_cdp_opt = results_dict[sigma_cdp.name()]
    covariance_cor_opt = results_dict[covariance_cor.name()]
    covariance_matrix_opt = sigma_cdp_opt * identity_matrix + covariance_cor_opt
    return covariance_matrix_opt


def _solve_problem(objective, constraints):
    """
    Solve the optimization problem.
    Parameters:
    - objective: cvxpy objective.
    - constraints: List of cvxpy constraints.
    Returns:
    - Dictionary of optimal variables, or None.
    """
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.MOSEK, verbose=False)
        if problem.status == cp.OPTIMAL:
            return {var.name(): var.value for var in problem.variables()}
        else:
            return None
    except Exception as e:
        warnings.warn(f"MOSEK failure: {e}")
        return None
