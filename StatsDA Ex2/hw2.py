# imports
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
import matplotlib.pyplot as plt


### Question 2 ###

def q2a(X, Y, Z):
    """
 
    Input:          
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].

    Explanation:
    We have 3 RV with respectively n,m and k possibles outcomes.
    The total number of possibles triples is n * m * k but have one constrain: The probabilities must sum up to 1
    It means that the number of free parameters is n * m * k - 1 
    
    Returns:
    The number of parameters that define the joint distribution of X, Y and Z.
    """
    n = X.shape[1]
    m = Y.shape[1]
    k = Z.shape[1]
    return (n * m * k) - 1

def q2b(X, Y, Z):
    """
 
    Input:          
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].

    Explanation:
    The 3 RV independents (and collectively independent as answered in the piazza) meaning that:
    P(X, Y, Z) = P(X) * P(Y) * P(Z)
    If a variable has r outcomes, its distribution has r-1 free params because the probabilities have to sum up to 1.
    Hence, the total number of parameters is n-1 + m-1 + k-1

    Returns:
    The number of parameters that define the joint distribution of X, Y and Z if we know that they are independent.
    """
    
    n = X.shape[1] - 1
    m = Y.shape[1] - 1
    k = Z.shape[1] - 1
    return n + m + k

def q2c(X, Y, Z):
    """
 
    Input:          
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    
    Explanations:
    When X and Y are conditionaly independent given Z we get: 
    P(X, Y, Z) = P(Z) * P(X | Z) * P(Y | Z).
    Z has k outcomes but there probabilities must sum up to 1 -> k-1 free parameters.
    For each values of k we have a full distribution over the values of X where for each z in Z we have n-1 possibles outcomes.
    Y follows the same logic as X.
    Hence we have in total k *(n-1) + k*(m-1) + (k-1).

    Returns:
    The number of parameters that define the joint distribution of X, Y and Z if we know that they are independent.
    """
    n = X.shape[1] - 1
    m = Y.shape[1] - 1
    k = Z.shape[1] 

    return k * n + k * m - k - 1


### Question 3 ###

def my_EM(
    mus=np.array([4.0, 9.0, np.nan]),
    sigmas=np.array([0.5, 0.5, 1.5]),
    ws=np.array([np.nan, 0.25, np.nan]),
    max_iter=200,
    tol=1e-6,
):
    """
 
    Input:
    - mus   : a numpy array: holds the initial guess for means of the Gaussians.
    - sigmas: a numpy array: holds the initial guess for std of the Gaussians.
    - ws    : a numpy array: holds the initial guess for weights of the Gaussians.

    - max_iteration: maximum number of iterations allowed. When reaching this number of iteration the loop stop and the last computed params are the result.
    - tol: Tolerance i.e. if the difference between the current params and the newly computed one are smaller than tol we reached a state of diminushing returN and the function stops.

    * The function should be generic and support any number of Gaussians.
      (you don't need to get this number as a parameter for the function. You can conclude it from the other parameters).

    Notes:
    - Please use np.nan to mark unknown parameters.

    Returns:
    The output of the EM algorithms (the GMM final parameters): mus, sigmas, ws.
    """

    data = pd.read_csv("GMD.csv", header=None).iloc[:, 1].to_numpy(dtype=float)
    x = data

    mus = np.array(mus, dtype=float)
    sigmas = np.array(sigmas, dtype=float)
    ws = np.array(ws, dtype=float)

    k = mus.size
    if sigmas.size != k or ws.size != k: # This means that the input is not valid. If you want to add an unknown parameter, please use np.nan 
        raise ValueError('mus, sigmas, ws must have the same length (numbers of Gaussians).')

    # masks: True = known/fixed, False = need to initialize
    mu_fixed    = ~np.isnan(mus) #return an array in which if a value is not nan we get False otherwise True ex:[np.nan, 0.25, np.nan] => [ True False  True]
    sigma_fixed = ~np.isnan(sigmas)
    w_fixed     = ~np.isnan(ws)

    # In this part we get the index of the missing values in mus i.e. the free value and we initialize them with the percentile of the data in GMD.csv 
    free_mu = np.flatnonzero(~mu_fixed) 
    if free_mu.size:
        mus[free_mu] = np.quantile(x, np.linspace(0.2, 0.8, free_mu.size))

    #Same idea as above. We fill free sigma with the standard deviation of x 
    sigmas[~sigma_fixed] = np.std(x)
    sigmas = np.maximum(sigmas, 1e-6)

    # we adds up the weights (if they were inialized correctly then there sum should be less or equal to 1), subtract 1 with this sum and divide the remaining equally among free weights
    if np.any(~w_fixed):
        remaining = max(1.0 - ws[w_fixed].sum(), 0.0)
        ws[~w_fixed] = remaining / (~w_fixed).sum()
    ws = ws / ws.sum()


    x_col = x[:, None]
    prev_ll = None
    log_ws = np.log(ws)
    update_mu = ~mu_fixed
    update_sigma = ~sigma_fixed
    update_w = ~w_fixed

    for _ in range(max_iter):
        ### E Steps:

        #compute responsibilities in log-space for numerical stability.
        # We could have used the standard pdf method but from our research it's better and more stable to use the exponential + logarithmic pdf method
        log_pdf = stats.norm.logpdf(x_col, loc=mus, scale=sigmas)
        log_prob = log_pdf + log_ws
        log_denom = logsumexp(log_prob, axis=1, keepdims=True)
        responsibilites = np.exp(log_prob - log_denom)

        # Effective component counts.
        Nk = responsibilites.sum(axis=0)

        ### M Steps:
        # Update only unknown means.
        if np.any(update_mu):
            mus[update_mu] = (responsibilites[:, update_mu] * x_col).sum(axis=0) / np.maximum(Nk[update_mu], 1e-12)

        # Update only unknown stds.
        if np.any(update_sigma):
            diff = x_col - mus
            sigmas[update_sigma] = np.sqrt(
                (responsibilites[:, update_sigma] * diff[:, update_sigma] ** 2).sum(axis=0)
                / np.maximum(Nk[update_sigma], 1e-12)
            )
            sigmas = np.clip(sigmas, 1e-6, None)

        # Update only unknown weights, renormalizing the free mass.
        if np.any(update_w):
            w_new = Nk / x.size
            fixed_sum = np.sum(ws[w_fixed])
            remaining = max(1.0 - fixed_sum, 0.0)
            free_sum = np.sum(w_new[update_w])
            if free_sum > 0:
                ws[update_w] = w_new[update_w] / free_sum * remaining
            else:
                ws[update_w] = remaining / np.sum(update_w)
            log_ws = np.log(ws)

        # Check log-likelihood improvement for convergence.
        ll = np.sum(log_denom)
        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return mus, sigmas, ws

def q3d(mus, sigmas, ws, n=1000):
    """
 
    Input:          
    - mus   : a numpy array: holds the means of the gaussians.
    - sigmas: a numpy array: holds the stds of the gaussians.
    - ws    : a numpy array: holds the weights of the gaussians.
    
    * The function should be generic and support any number of Gaussians.
      (you don't need to get this number as a parameter for the function. You can conclude it from the other parameters).
    
    Returns:
    The generated data.
    """
    mus = np.array(mus, dtype=float)
    sigmas = np.array(sigmas, dtype=float)
    ws = np.array(ws, dtype=float)

    k = mus.size
    if sigmas.size != k or ws.size != k:
        raise ValueError('mus, sigmas, ws must have the same length (numbers of Gaussians).')

    ws = ws / ws.sum()
    # Sample component indices, then draw from the corresponding Gaussians.
    comps = np.random.choice(k, size=n, p=ws) # This help us assign each element to a gaussian
    samples = mus[comps] + sigmas[comps] * np.random.randn(n) # This generate the random number for each gaussian
    return samples


### Question 4 ###

def q4a(mu=75000, sigma=37500, salary=50000):
    """

    Input:
    - mu   : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma: The std of the annual salaries of employees in a large Randomistan company.
    The annual salary of employees in a large Randomistan company are approximately normally distributed.

    Returns:
    The percent of people earn less than 'salary'.
    """
    # P(X < salary) where X ~ N(mu, sigma^2)
    # Use cumulative distribution function (CDF)
    prob = stats.norm.cdf(salary, loc=mu, scale=sigma)
    return prob * 100  # Convert to percentage

def q4b(mu=75000, sigma=37500, min_salary=45000, max_salary=65000):
    """

    Input:
    - mu   : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma: The std of the annual salaries of employees in a large Randomistan company.
    The annual salary of employees in a large Randomistan company are approximately normally distributed.

    Returns:
    The percent of people earn between 'min_salary' and 'max_salary'.
    """
    # P(min_salary < X < max_salary) = P(X < max_salary) - P(X < min_salary)
    prob = stats.norm.cdf(max_salary, loc=mu, scale=sigma) - stats.norm.cdf(min_salary, loc=mu, scale=sigma)
    return prob * 100  # Convert to percentage

def q4c(mu=75000, sigma=37500, salary=85000):
    """

    Input:
    - mu   : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma: The std of the annual salaries of employees in a large Randomistan company.
    The annual salary of employees in a large Randomistan company are approximately normally distributed.

    Returns:
    The percent of people earn more than 'salary'.
    """
    # P(X > salary) = 1 - P(X <= salary)
    prob = 1 - stats.norm.cdf(salary, loc=mu, scale=sigma)
    return prob * 100  # Convert to percentage

def q4d(mu=75000, sigma=37500, salary=140000, n_employees=1000):
    """

    Input:
    - mu         : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma      : The std of the annual salaries of employees in a large Randomistan company.
    - n_employees: The number of employees in the company
    The annual salary of employees in a large Randomistan company are approximately normally distributed.

    Returns:
    The number of employees in the company that you expect to earn more than 'salary'.
    """
    # Expected number = n_employees * P(X > salary)
    prob = 1 - stats.norm.cdf(salary, loc=mu, scale=sigma)
    expected_count = n_employees * prob
    return expected_count


### Question 5 ###

def CC_Expected(N=10):
    """

    Input:
    - N: Number of different coupons.

    Returns:
    E(T_N)
    """
    # T_N = sum of T_i where T_i ~ Geometric(p_i)
    # p_i = (N - i + 1) / N is the probability of getting a new coupon when you have i-1 distinct coupons
    # E(T_i) = 1/p_i = N / (N - i + 1)
    # E(T_N) = sum_{i=1}^{N} N / (N - i + 1) = N * sum_{j=1}^{N} 1/j = N * H_N (harmonic number)

    expected = N * np.sum(1.0 / np.arange(1, N + 1))
    return expected

def CC_Variance(N=10):
    """

    Input:
    - N: Number of different coupons.

    Returns:
    V(T_N)
    """
    # T_i are independent, so V(T_N) = sum of V(T_i)
    # For geometric distribution with success probability p: V(T) = (1-p) / p^2
    # p_i = (N - i + 1) / N
    # V(T_i) = (1 - p_i) / p_i^2 = (N - (N - i + 1)) / ((N - i + 1) / N)^2
    #        = (i - 1) / ((N - i + 1)^2 / N^2)
    #        = N^2 * (i - 1) / (N - i + 1)^2

    variance = 0.0
    for i in range(1, N + 1):
        p_i = (N - i + 1) / N
        variance += (1 - p_i) / (p_i ** 2)

    return variance

def CC_T_Steps(N=10, n_steps=30):
    """

    Input:
    - N: Number of different coupons.
    - n_steps: Number of steps.

    Returns:
    The probability that T_N > n_steps
    """
    # Use dynamic programming to compute P(T_N <= n_steps)
    # State: dp[t][k] = probability of having exactly k distinct coupons after t draws (and not yet finished)
    # finished[t] = cumulative probability of finishing by time t

    # Initialize: dp[0][0] = 1 (no coupons collected at time 0)
    dp = np.zeros((n_steps + 1, N + 1))
    dp[0][0] = 1.0

    finished_cumulative = 0.0  # Track cumulative probability of having finished

    # Transition: at time t, if we have k distinct coupons:
    # - Probability k/N of drawing an existing coupon -> stay at k
    # - Probability (N-k)/N of drawing a new coupon -> move to k+1
    for t in range(n_steps):
        for k in range(N):  # Only k < N since k=N means we're done
            if dp[t][k] == 0:
                continue
            # Draw existing coupon (stay at k)
            dp[t + 1][k] += dp[t][k] * (k / N)
            # Draw new coupon (move to k+1)
            if k + 1 < N:
                dp[t + 1][k + 1] += dp[t][k] * ((N - k) / N)
            else:  # k + 1 == N, we just finished!
                finished_cumulative += dp[t][k] * ((N - k) / N)

    # P(T_N <= n_steps) = cumulative probability of finishing by time n_steps
    return 1.0 - finished_cumulative  # P(T_N > n_steps)

def CC_S_Steps(N=10, n_steps=30):
    """

    Input:
    - N: Number of different coupons.
    - n_steps: Number of steps.

    Returns:
    The probability that S_N > n_steps
    """
    # S_N = time to collect all N coupons at least twice
    # State: (k0, k1, k2) where k0 = coupons with 0 copies, k1 = 1 copy, k2 = 2+ copies
    # Constraint: k0 + k1 + k2 = N
    # We only need to track k1 and k2 since k0 = N - k1 - k2
    # Goal: reach state where k2 = N (all coupons collected at least twice)

    # Use dp[t][(k1, k2)] = probability of state (k1, k2) at time t (not yet finished)
    from collections import defaultdict

    # Initialize: at t=0, all N coupons have 0 copies
    dp = [defaultdict(float) for _ in range(n_steps + 1)]
    dp[0][(0, 0)] = 1.0  # k1=0, k2=0, so k0=N

    finished_cumulative = 0.0  # Track cumulative probability of finishing

    for t in range(n_steps):
        for (k1, k2), prob in dp[t].items():
            if prob == 0:
                continue

            k0 = N - k1 - k2  # coupons with 0 copies

            # Draw a coupon with 0 copies (probability k0/N): k0 -> k0-1, k1 -> k1+1
            if k0 > 0:
                dp[t + 1][(k1 + 1, k2)] += prob * (k0 / N)

            # Draw a coupon with 1 copy (probability k1/N): k1 -> k1-1, k2 -> k2+1
            if k1 > 0:
                new_k2 = k2 + 1
                if new_k2 < N:
                    dp[t + 1][(k1 - 1, new_k2)] += prob * (k1 / N)
                else:  # new_k2 == N, we just finished!
                    finished_cumulative += prob * (k1 / N)

            # Draw a coupon with 2+ copies (probability k2/N): stay at (k1, k2)
            if k2 > 0:
                dp[t + 1][(k1, k2)] += prob * (k2 / N)

    # P(S_N <= n_steps) = cumulative probability of finishing by time n_steps
    return 1.0 - finished_cumulative  # P(S_N > n_steps)

