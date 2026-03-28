# imports 
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import nbinom,binom,multinomial
import matplotlib.pyplot as plt
import math

### Question 1 ###

#p_target - The desired probability of detecting at least one defective item
#prob_defective - The defective rate of the production line (probability that any single item is defective)
def find_sample_size_binom(p_target = 0.85, prob_defective = 0.03):
    """
    Using Binom to returns the minimal number of samples required to have requested probability of receiving 
    at least x defective products from a production line with a defective rate.
    """
    #With Scipy
    #For each sample size n, it calculates: P(X ≥ 1) where X is the number of defective items
    #Starts with n=1 sample and incrementally increases n until the calculated probability meets or exceeds p_target
    n=1
    while True:
        p = 1-binom.cdf(0,n,prob_defective)
        if p>=p_target:
            return n
        n+=1
    

# r - The number of defective items (successes) you want to find
# p_target - The desired probability threshold
# prob_defective - The defective rate per item

def find_sample_size_nbinom(r=1, p_target=0.85, prob_defective=0.03):
    """
    Using NBinom to returns the minimal number of samples required to have requested probability of receiving 
    at least x defective products from a production line with a defective rate.
    """
    n = r #minimum possible, since we need at least r trials to get r defectives
    while True:
        k = n-r #number of failures before getting r successes
        p = nbinom.cdf(k,r,prob_defective)
        if p>=p_target:
            return n
        n+=1

#Compares two different scenarios of defect detection by calculating the required sample sizes for each case using the negative binomial distribution.
def compare_q1(r1 = 5,p1_target=0.9,p1=0.10,r2=15,p2_target=0.90,p2=0.3):
    n1 = find_sample_size_nbinom(r1,p1_target,p1)
    print("The number of independant samples for the first case is ",n1)
    n2 = find_sample_size_nbinom(r2,p2_target,p2)
    print("The number of independant samples for the second case is ",n2)
    return n1,n2


#Finds the sample size n where both scenarios have approximately equal probabilities of detecting their target number of defects
#In this case, rather than using the nbinom we will instead use the regular binom function for each steps to compares the 2 cases 
def same_prob(r1 = 5,p1=0.10,r2=15,p2=0.3):
    n=1
    while True:
        #Rather than calulating all the possible case, we would rather computes all the cases under n (less steps)
        prob1 = 1 - binom.cdf(r1-1,n,p1)
        prob2 = 1 - binom.cdf(r2-1,n,p2)
        if prob1>0 and prob2>0:
            #Considers them equal if they differ by less than 0.01 (1%)
            if np.isclose(prob1,prob2,atol=1e-2):
                return n
        n+=1



### Question 2 ###

# Empirical approach
def empirical_centralized_third_moment(n=20, p=[0.2, 0.1, 0.1, 0.1, 0.2, 0.3], k=100, seed=None):
    """
    Create k experiments where X is sampled. Calculate the empirical centralized third moment of Y based 
    on your k experiments.
    """
    if seed is not None:
        np.random.seed(seed)
    X = multinomial.rvs(n,p,size=k,random_state=seed)
    Y = X[:, 1] + X[:, 2] + X[:, 3] #only taking the second,third and forth columns
    Y_mean = np.mean(Y)
    empirical_moment = np.mean((Y-Y_mean)**3)
    
    return empirical_moment

def class_moment_scipy_formula(n=20,p=0.3,k=100):
    # From the first question we know that we are only interested in y = x1+x2+x3 so we define 
    # p =  0.3 as the default value, n stays the same here

    mean, var, skew = binom.stats(n,p,moments='mvs')
    cent_third_moment = skew * (var)**(3/2) # according to the formula
    return cent_third_moment

# Theoretical approach
def class_moment(n=20,p=0.3,k=100):
    # From the first question we know that we are only interested in y = x1+x2+x3 so we define 
    # p =  0.3 as the default value, n stays the same here

    mean = n*p
    mu3 = n*p*(1-p)*(1-2*p)
    return mu3

# Visualization - Compare empirical vs theoretical
def plot_moments(n=20, p=[0.2, 0.1, 0.1, 0.1, 0.2, 0.3], k=100, seed=None, bins=30, num_exp =1000):

    moments = np.empty(num_exp,dtype=float)
    p1 = p[1]+p[2]+p[3] 
    for i in range(num_exp):
        moments[i] = empirical_centralized_third_moment(n,p,k,seed)
    

    third_moment = class_moment(n,p1)
    #third_moment_scipy = class_moment_scipy_formula(n,p1)
    fig, ax = plt.subplots()
    ax.hist(moments,bins=bins)
    ax.axvline(third_moment, color="red", linestyle="--", linewidth=2, label=f"class μ₃ = {third_moment:.3f}")
    #ax.axvline(third_moment_scipy, color="green", linestyle="--", linewidth=2, label=f"class μ₃ cf = {third_moment_scipy:.3f}")
    ax.set_xlabel("Cent. Third moment of Y")
    ax.set_ylabel("Frequency")
    ax.set_title("Centralized Third moment of Y after 1000 experiments")
    ax.legend()
    plt.tight_layout()

    dist_var = np.var(moments,ddof=1) # for sample variance, otherwise if you want population variation please enter ddof = 0
    return dist_var

# Improving accuracy - Get tighter distribution around the true value
# You can use a larger K but it will take longer to run
def plot_moments_smaller_variance(n=20, p=[0.2, 0.1, 0.1, 0.1, 0.2, 0.3], k=10000, seed=None, bins=30, num_exp =1000):
    print("We increased the value K, i.e. we have more samples per experiments, which will give us a tighter distribution and smaller variance")
    moments = np.empty(num_exp,dtype=float)
    p1 = p[1]+p[2]+p[3]
    for i in range(num_exp):
        moments[i] = empirical_centralized_third_moment(n,p,k,seed)
    

    #third_moment_scipy = class_moment_scipy_formula(n,p1)
    third_moment = class_moment(n,p1)
    fig, ax = plt.subplots()
    ax.hist(moments,bins=bins)
    #ax.axvline(third_moment_scipy, color="green", linestyle="--", linewidth=2, label=f"class μ₃ = {third_moment_scipy:.3f}")
    ax.axvline(third_moment, color="red", linestyle="--", linewidth=2, label=f"class μ₃ cf = {third_moment:.3f}")
    ax.set_xlabel("Cent. Third moment of Y")
    ax.set_ylabel("Frequency")
    ax.set_title("Centralized Third moment of Y after 1000 experiments")
    ax.legend()
    plt.tight_layout()
    ###
    dist_var = np.var(moments,ddof=1) # for sample variance, otherwise if you want population variation please enter ddof = 0
    return dist_var



### Question 3 ###

def NFoldConv(P, n):
    """
    Calculating the distribution, Q, of the sum of n independent repeats of random variables, 
    each of which has the distribution P.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    - n: An integer.

    Returns:
    - Q: 2d numpy array: [[values], [probabilities]].
    """
    Q = P.copy()

    for i in range(n-1):
        Q = np.convolve(Q,P)
    # Normalize to ensure probabilities sum to 1
    Q = Q/Q.sum()
    return Q
    
def plot_dist(P):
    """
    Ploting the distribution P using barplot.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    """
    
    P = np.asarray(P, dtype=float)
    x_vals = np.arange(len(P))

    plt.figure()
    plt.bar(x_vals, P, align='center')
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.title("Distribution P")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

'''
#Script to test the functions in Q3
p = 0.3
P = np.array([1 - p, p])   # X in {0,1}
print("Base distribution P (Bernoulli):")
print(P)
plot_dist(P)
# Sum of n iid Bernoulli(p) -> Binomial(n, p)
n = 5
Q = NFoldConv(P, n)
print(f"\nNFoldConv(P, n={n}) gives:")
print(Q)
plot_dist(Q)
# Compare to the exact Binomial pmf
k_values = np.arange(len(Q))  # 0,...,n
binom_pmf = binom.pmf(k_values, n, p)
print("\nExact Binomial(n, p) pmf:")
print(binom_pmf)
print("\nDifference (Q - Binomial pmf):")
print(Q - binom_pmf)'''




### Question 4 ###

def evenBinom(n, p):
    """
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p).

    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """
    # Calculate P(X is even) by summing probabilities for all even values
    # P(X is even) = P(X=0) + P(X=2) + P(X=4) + ... + P(X=n or n-1)
    prob = 0
    for k in range(0, n+1, 2):  # Iterate over even values: 0, 2, 4, ...
        prob += binom.pmf(k, n, p)

    return prob


# This formula is much more efficient than the iterative approach in evenBinom - 
# it computes the result in O(1) time instead of O(n)!
def evenBinomFormula(n, p):
    """
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p) Using a closed-form formula.
    It should also print the proof for the formula.

    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """
    
    # Print the proof for the formula
    print("=" * 70)
    print("PROOF: Closed-form formula for P(X is even), X ~ Binom(n, p)")
    print("=" * 70)
    print("\nLet X ~ Binom(n, p), where q = 1 - p.")
    print("\nUsing the binomial theorem:")
    print("  (p + q)^n = Σ(k=0 to n) C(n,k) * p^k * q^(n-k)")
    print("  (q - p)^n = Σ(k=0 to n) C(n,k) * (-p)^k * q^(n-k)")
    print("            = Σ(k=0 to n) C(n,k) * p^k * q^(n-k) * (-1)^k")
    print("\nWhen k is even, (-1)^k = +1")
    print("When k is odd,  (-1)^k = -1")
    print("\nAdding the two expansions:")
    print("  (p + q)^n + (q - p)^n = 2 * Σ(k even) C(n,k) * p^k * q^(n-k)")
    print("\nSince p + q = 1:")
    print("  1 + (1 - 2p)^n = 2 * P(X is even)")
    print("\nTherefore:")
    print("  P(X is even) = [1 + (1 - 2p)^n] / 2")
    print("=" * 70)

    # Calculate using the closed-form formula
    prob = (1 + (1 - 2*p)**n) / 2

    return prob




### Question 5 ###

def three_RV(values, joint_probs):
    """

    Input:
    - values: 3d numpy array of tuples: all the value combinations of X, Y, and Z
      Each tuple has the form (x_i, y_j, z_k) representing the i, j, and k values of X, Y, and Z, respectively
    - joint_probs: 3d numpy array: joint probability of X, Y, and Z
      The marginal distribution of each RV can be calculated from the joint distribution

    Returns:
    - v: The variance of X + Y + Z. (you cannot create the RV U = X + Y + Z)
    """
    # Flatten values and joint_probs for easier computation
    values_flat = values.reshape(-1, 3)  # Shape: (N, 3) where N is total number of outcomes
    probs_flat = joint_probs.flatten()    # Shape: (N,)

    # Extract X, Y, Z values
    X_vals = values_flat[:, 0]
    Y_vals = values_flat[:, 1]
    Z_vals = values_flat[:, 2]

    # Calculate expected values E[X], E[Y], E[Z]
    E_X = np.sum(X_vals * probs_flat)
    E_Y = np.sum(Y_vals * probs_flat)
    E_Z = np.sum(Z_vals * probs_flat)

    # Calculate E[X^2], E[Y^2], E[Z^2]
    E_X2 = np.sum(X_vals**2 * probs_flat)
    E_Y2 = np.sum(Y_vals**2 * probs_flat)
    E_Z2 = np.sum(Z_vals**2 * probs_flat)

    # Calculate variances: Var(X) = E[X^2] - E[X]^2
    Var_X = E_X2 - E_X**2
    Var_Y = E_Y2 - E_Y**2
    Var_Z = E_Z2 - E_Z**2

    # Calculate E[XY], E[XZ], E[YZ] for covariances
    E_XY = np.sum(X_vals * Y_vals * probs_flat)
    E_XZ = np.sum(X_vals * Z_vals * probs_flat)
    E_YZ = np.sum(Y_vals * Z_vals * probs_flat)

    # Calculate covariances: Cov(X,Y) = E[XY] - E[X]E[Y]
    Cov_XY = E_XY - E_X * E_Y
    Cov_XZ = E_XZ - E_X * E_Z
    Cov_YZ = E_YZ - E_Y * E_Z

    # Variance of X + Y + Z = Var(X) + Var(Y) + Var(Z) + 2*Cov(X,Y) + 2*Cov(X,Z) + 2*Cov(Y,Z)
    v = Var_X + Var_Y + Var_Z + 2*Cov_XY + 2*Cov_XZ + 2*Cov_YZ

    # Print the formula and explanation
    print("=" * 70)
    print("Variance of X + Y + Z")
    print("=" * 70)
    print("\nFormula:")
    print("Var(X + Y + Z) = Var(X) + Var(Y) + Var(Z)")
    print("                 + 2·Cov(X,Y) + 2·Cov(X,Z) + 2·Cov(Y,Z)")
    print("\nCalculated values:")
    print(f"  Var(X) = {Var_X:.6f}")
    print(f"  Var(Y) = {Var_Y:.6f}")
    print(f"  Var(Z) = {Var_Z:.6f}")
    print(f"  Cov(X,Y) = {Cov_XY:.6f}")
    print(f"  Cov(X,Z) = {Cov_XZ:.6f}")
    print(f"  Cov(Y,Z) = {Cov_YZ:.6f}")
    print(f"\nVar(X + Y + Z) = {v:.6f}")
    print("=" * 70)

    return v

def three_RV_pairwise_independent(values, joint_probs):
    """

    Input:
    - values: 3d numpy array of tuples: all the value combinations of X, Y, and Z
      Each tuple has the form (x_i, y_j, z_k) representing the i, j, and k values of X, Y, and Z, respectively
    - joint_probs: 3d numpy array: joint probability of X, Y, and Z
      The marginal distribution of each RV can be calculated from the joint distribution

    Returns:
    - v: The variance of X + Y + Z. (you cannot create the RV U = X + Y + Z)
    """
    # Flatten values and joint_probs for easier computation
    values_flat = values.reshape(-1, 3)  # Shape: (N, 3) where N is total number of outcomes
    probs_flat = joint_probs.flatten()    # Shape: (N,)

    # Extract X, Y, Z values
    X_vals = values_flat[:, 0]
    Y_vals = values_flat[:, 1]
    Z_vals = values_flat[:, 2]

    # Calculate expected values E[X], E[Y], E[Z]
    E_X = np.sum(X_vals * probs_flat)
    E_Y = np.sum(Y_vals * probs_flat)
    E_Z = np.sum(Z_vals * probs_flat)

    # Calculate E[X^2], E[Y^2], E[Z^2]
    E_X2 = np.sum(X_vals**2 * probs_flat)
    E_Y2 = np.sum(Y_vals**2 * probs_flat)
    E_Z2 = np.sum(Z_vals**2 * probs_flat)

    # Calculate variances: Var(X) = E[X^2] - E[X]^2
    Var_X = E_X2 - E_X**2
    Var_Y = E_Y2 - E_Y**2
    Var_Z = E_Z2 - E_Z**2

    # For pairwise independent RVs, all covariances are zero:
    # Cov(X,Y) = 0, Cov(X,Z) = 0, Cov(Y,Z) = 0
    # Therefore: Var(X + Y + Z) = Var(X) + Var(Y) + Var(Z)
    v = Var_X + Var_Y + Var_Z

    # Print the explanation
    print("=" * 70)
    print("Variance of X + Y + Z (Pairwise Independent Case)")
    print("=" * 70)
    print("\nWhen X, Y, Z are pairwise independent:")
    print("  - Cov(X,Y) = 0")
    print("  - Cov(X,Z) = 0")
    print("  - Cov(Y,Z) = 0")
    print("\nThis simplifies the variance formula to:")
    print("  Var(X + Y + Z) = Var(X) + Var(Y) + Var(Z)")
    print("\nCalculated values:")
    print(f"  Var(X) = {Var_X:.6f}")
    print(f"  Var(Y) = {Var_Y:.6f}")
    print(f"  Var(Z) = {Var_Z:.6f}")
    print(f"\nVar(X + Y + Z) = {v:.6f}")
    print("=" * 70)

    return v

def is_pairwise_collectively(values, joint_probs):
    """

    Input:
    - values: 3d numpy array of tuples: all the value combinations of X, Y, and Z
      Each tuple has the form (x_i, y_j, z_k) representing the i, j, and k values of X, Y, and Z, respectively
    - joint_probs: 3d numpy array: joint probability of X, Y, and Z
      The marginal distribution of each RV can be calculated from the joint distribution

    Returns:
    TRUE or FALSE
    """
    # Calculate marginal distributions
    # P(X=x) by summing over all y and z
    P_X = np.sum(joint_probs, axis=(1, 2))

    # P(Y=y) by summing over all x and z
    P_Y = np.sum(joint_probs, axis=(0, 2))

    # P(Z=z) by summing over all x and y
    P_Z = np.sum(joint_probs, axis=(0, 1))

    # Check collective independence: P(X,Y,Z) = P(X) * P(Y) * P(Z)
    # Create the product of marginals
    # Use broadcasting to create a 3D array of P(X=x_i) * P(Y=y_j) * P(Z=z_k)
    product_of_marginals = P_X[:, np.newaxis, np.newaxis] * P_Y[np.newaxis, :, np.newaxis] * P_Z[np.newaxis, np.newaxis, :]

    # Check if joint distribution equals product of marginals (within numerical tolerance)
    is_collectively_independent = np.allclose(joint_probs, product_of_marginals, atol=1e-10)

    # Print explanation
    print("=" * 70)
    print("Checking Collective Independence")
    print("=" * 70)
    print("\nPairwise independence does NOT necessarily imply collective independence.")
    print("\nFor collective (mutual) independence, we need:")
    print("  P(X=x, Y=y, Z=z) = P(X=x) · P(Y=y) · P(Z=z)  for ALL x, y, z")
    print("\nChecking if this condition holds...")

    if is_collectively_independent:
        print("\nResult: TRUE - The RVs are collectively independent")
        print("(In this case, pairwise independence implies collective independence)")
    else:
        print("\nResult: FALSE - The RVs are NOT collectively independent")
        print("(Pairwise independence does not imply collective independence)")

        # Show an example where they differ
        max_diff = np.max(np.abs(joint_probs - product_of_marginals))
        print(f"\nMaximum difference between joint and product of marginals: {max_diff:.6e}")

    print("=" * 70)

    return is_collectively_independent





### Question 6 ###

def expectedC(n, p):
    """
    The program outputs the expected value of the RV C as defined in the notebook.

    Given:
    - Ω = {0,1}^n (all binary strings of length n)
    - P is induced by independently tossing a p-coin n times
    - W(ω) = number of 1s in ω
    - C(ω) = |{ζ : W(ζ) = W(ω)}| (count of strings with same number of 1s as ω)

    For a string with k ones, C = C(n,k) = n choose k

    E[C] = Σ_{k=0}^n P(W=k) · C(n,k)
         = Σ_{k=0}^n [C(n,k) · p^k · (1-p)^(n-k)] · C(n,k)
         = Σ_{k=0}^n C(n,k)^2 · p^k · (1-p)^(n-k)
    """

    # Calculate E[C] = Σ_{k=0}^n C(n,k)^2 · p^k · (1-p)^(n-k)
    expected_value = 0.0

    for k in range(n + 1):
        # C(n,k) = binomial coefficient
        binom_coef = math.comb(n, k)

        # P(W=k) = C(n,k) · p^k · (1-p)^(n-k)
        prob_W_k = binom.pmf(k, n, p)



        # C(ω) for a string with k ones is C(n,k)
        C_value = binom_coef

        # Add to expected value: P(W=k) · C(ω)
        expected_value += prob_W_k * C_value

    # Print explanation
    print("=" * 70)
    print("Expected Value of C")
    print("=" * 70)
    print(f"\nParameters: n={n}, p={p}")
    print("\nDefinition:")
    print("  Ω = {0,1}^n (all binary strings of length n)")
    print("  W(ω) = number of 1s in ω")
    print("  C(ω) = count of strings with same number of 1s as ω")
    print("\nFor a string with k ones: C(ω) = C(n,k) = n choose k")
    print("\nFormula:")
    print("  E[C] = Σ_{k=0}^n P(W=k) · C(n,k)")
    print("       = Σ_{k=0}^n C(n,k)^2 · p^k · (1-p)^(n-k)")
    print(f"\nE[C] = {expected_value:.6f}")
    print("=" * 70)

    return expected_value
