#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("All models are wrong, but some are useful.")


# In[7]:


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import pymc3 as pm
import pystan
plt.style.use('seaborn-darkgrid')


# In[8]:


##############################################
# prioir x likelihood = posterior
##############################################

success=6

tosses=9

# define grid
grid_points=100

# define grid
p_grid = np.linspace(0, 1, grid_points)

# compute likelihood at each point in the grid
likelihood = stats.binom.pmf(success, tosses, p_grid)


# In[9]:


def computePosterior(likelihood, prior):
    
    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # standardize posterior
    posterior = unstd_posterior / unstd_posterior.sum()
    
    plt.figure(figsize=(17, 3))
    ax1 = plt.subplot(131)
    ax1.set_title("Prior")
    plt.plot(p_grid, prior)

    ax2 = plt.subplot(132)
    ax2.set_title("Likelihood")
    plt.plot(p_grid, likelihood)

    ax3 = plt.subplot(133)
    ax3.set_title("Posterior")
    plt.plot(p_grid, posterior)
    plt.show()
    
    return posterior
    


# In[10]:


prior1 = np.repeat(1, grid_points)  
posterior1 = computePosterior(likelihood, prior1)


# In[11]:


prior2 = (p_grid >= 0.5).astype(int)
posterior2 = computePosterior(likelihood, prior2)


# In[12]:


prior3 = np.exp(- 5 * abs(p_grid - 0.5)) 
posterior3 = computePosterior(likelihood, prior3)


# ## The Monte Carlo method

# In[ ]:


##############################################
# the Monte Carlo method - compute pi
##############################################


# In[13]:


def in_circle(x, y, r):
    return math.sqrt(x **2 + y**2) <= r**2


# In[14]:


def approx_pi(r, n):
    
    xs, ys, cols = [], [], []
    
    count = 0
    
    for i in range(n):
        x = np.random.uniform(0,r,1)
        y = np.random.uniform(0,r,1)
        xs.append(x)
        ys.append(y)

        if in_circle(x, y, r):
            count += 1
            cols.append("red")
        else:
            cols.append("steelblue")
            
    pi_appr = round(4 * count/n, 3)
    
    plt.figure(figsize=(2, 2))
    plt.scatter(xs, ys, c = cols, s=2)
    plt.title("pi (approximately) = " + str(pi_appr))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    return pi_appr


# In[15]:


r = 1

for n in 5*10**np.array([1,2,3]):
    approx_pi(r, n)


# ## Monte Carlo integration

# In[1]:


##############################################
# the Monte Carlo method - integration
##############################################


# In[16]:


x = np.linspace(0, 1, 100)
plt.plot(x, np.exp(x));
pts = np.random.uniform(0,1,(100, 2))
pts[:, 1] *= np.e

cols = ['steelblue'] * 100
for i in range(100):
    if pts[i,1] > np.exp(pts[i,0]):     # acceptance / rejection step
        cols[i] = 'red'
    
plt.scatter(pts[:, 0], pts[:, 1], c = cols)
plt.xlim([0,1])
plt.ylim([0, np.e]);


# In[17]:


# Monte Carlo approximation

for n in 10**np.array([1,2,3,4,5,6,7,8]):
    pts = np.random.uniform(0, 1, (n, 2))
    pts[:, 1] *= np.e
    count = np.sum(pts[:, 1] < np.exp(pts[:, 0]))
    volume = np.e * 1 # volume of region
    sol = (volume * count)/n    
    print('%10d %.6f' % (n, sol))


# ## Mandatory coin tossing example

# In[ ]:


##############################################
# coin tossing
##############################################


# In[18]:


n = 10
h = 6
p = h/n
p


# In[19]:


a, b = 10, 10                   # hyperparameters
prior = stats.beta(a, b)        # prior
post = stats.beta(h+a, n-h+b)   # posterior


# In[20]:


def beta_binomial(n, h, a, b):
    # frequentist
    p = h/n
    rv = stats.binom(n, p)
    mu = rv.mean()
    
    # Bayesian
    prior = stats.beta(a, b)
    post = stats.beta(h+a, n-h+b)
    
    thetas = np.linspace(0, 1, 200)
    plt.figure(figsize=(8, 6))
    plt.plot(thetas, prior.pdf(thetas), label='Prior', c='blue')
    plt.plot(thetas, post.pdf(thetas), label='Posterior', c='red')
    plt.plot(thetas, n*stats.binom(n, thetas).pmf(h), label='Likelihood', c='green')
    plt.axvline((h+a-1)/(n+a+b-2), c='red', linestyle='dashed', alpha=0.4, label='MAP')
    plt.axvline(mu/n, c='green', linestyle='dashed', alpha=0.4, label='MLE')
    plt.xlim([0, 1])
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.ylabel('Density', fontsize=16)
    plt.legend();


# In[21]:


beta_binomial(100, 80, 10, 10)


# In[22]:


beta_binomial(4, 3, 10, 10)


# In[23]:


beta_binomial(4, 3, 2, 2)


# In[24]:


beta_binomial(4, 3, 1, 1)


# ## Metropolis-Hastings random walk algorithm

# In[ ]:


##############################################
# Metropolis-Hastings
##############################################


# In[19]:


def target(likelihood, prior, n, h, theta):
    if theta < 0 or theta > 1:
        return 0
    else:
        return likelihood(n, theta).pmf(h)*prior.pdf(theta)


# In[20]:


n = 100
h = 61
a = 10
b = 10
likelihood = stats.binom
prior = stats.beta(a, b)
sigma = 0.3


# In[21]:


naccept = 0
theta = 0.1
niters = 10000

samples = np.zeros(niters+1)
samples[0] = theta

for i in range(niters):
    theta_p = theta + stats.norm(0, sigma).rvs()
    rho = min(1, target(likelihood, prior, n, h, theta_p)/target(likelihood, prior, n, h, theta ))
    u = np.random.uniform()
    if u < rho:
        naccept += 1
        theta = theta_p
    samples[i+1] = theta


# In[22]:


nmcmc = len(samples)//2
print("Portion of accepted steps = " + str(naccept/niters))


# In[23]:


post = stats.beta(h+a, n-h+b)
thetas = np.linspace(0, 1, 200)

plt.figure(figsize=(8, 6))
plt.hist(samples[nmcmc:], 20, histtype='step', normed=True, linewidth=1, label='Distribution of posterior samples');
plt.hist(prior.rvs(nmcmc), 40, histtype='step', normed=True, linewidth=1, label='Distribution of prior samples');
plt.plot(thetas, post.pdf(thetas), c='red', linestyle='--', alpha=0.5, label='True posterior')
plt.xlim([0,1]);
plt.legend(loc='best');


# ## Convergence diagnostics

# In[24]:


def mh_coin(niters, n, h, theta, likelihood, prior, sigma):
    samples = [theta]
    while len(samples) < niters:
        theta_p = theta + stats.norm(0, sigma).rvs()
        rho = min(1, target(likelihood, prior, n, h, theta_p)/target(likelihood, prior, n, h, theta ))
        u = np.random.uniform()
        if u < rho:
            theta = theta_p
        samples.append(theta)
        
    return samples


# In[25]:


n = 100
h = 61
lik = stats.binom
prior = stats.beta(a, b)
sigma = 0.05
niters = 100


# In[26]:


chains = [mh_coin(niters, n, h, theta, likelihood, prior, sigma) for theta in np.arange(0.1, 1, 0.2)]


# In[27]:


plt.figure(figsize=(8, 6))

for chain in chains:
    plt.plot(chain, '-o')
    
plt.xlim([0, niters])
plt.ylim([0, 1]);


# ## PyMC3
# 
# https://docs.pymc.io/
# 

# In[31]:


with pm.Model() as model:
    # Model definition
    pass


# In[32]:


with pm.Model():
    x = pm.Normal('x', mu=0, sd=1)


# ## Coin tossing problem - PyMC3

# In[34]:


n = 100    # number of trials
h = 61     # number of successes
#alpha = 2  # hyperparameters
#beta = 2

niter = 1000


# In[35]:


get_ipython().run_cell_magic('time', '', "with pm.Model() as model: \n    # prior\n    p = pm.Beta('p', alpha=2, beta=2)\n\n    # likelihood\n    y = pm.Binomial('y', n=n, p=p, observed=h)\n\n    # inference\n    start = pm.find_MAP()  # Use MAP estimate (optimization) as the initial state for MCMC\n    step = pm.Metropolis() # Have a choice of samplers\n    \n    trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)")


# In[10]:


alpha = 2  # hyperparameters
beta = 2
plt.hist(trace['p'], 15, histtype='step', normed=True, label='posterior');
x = np.linspace(0, 1, 100)
plt.plot(x, stats.beta.pdf(x, alpha, beta), label='prior');
plt.legend(loc='best');


# In[36]:


# traceplot
pm.traceplot(trace)
pass


# In[37]:


# extract values
p_samps_pymc = trace.get_values('p', chains = [0,1,2,3])
fig, ax = plt.subplots()
ax.plot(p_samps_pymc)
pass


# In[38]:


# summary
pm.summary(trace).round(2)


# In[39]:


# diagnostics: Gelman-Rubin
print(pm.diagnostics.gelman_rubin(trace))


# In[40]:


# diagnostics: n effective
print(pm.diagnostics.effective_n(trace))


# In[41]:


pm.forestplot(trace);


# In[42]:


pm.plot_posterior(trace);


# PyMC3, offers a variety of other samplers, found in pm.step_methods.

# In[43]:


list(filter(lambda x: x[0].isupper(), dir(pm.step_methods)))


# ## Estimating parameters of the normal distribution

# 
# $$y ∼ N(\mu, \sigma^2)$$

# ### Data generation

# In[ ]:


##############################################
# normal distribution
##############################################


# In[60]:


N = 100
niter = 20000
x = np.random.normal(0,1,N)
plt.hist(x)
pass


# In[61]:


# assume known variance, unknown mean

with pm.Model() as model1:
    # prior
    mean = pm.Normal('mu', mu=0, sd=0.5)
    # likelihood
    obs = pm.Normal('obs', mu=mean, sd=1, observed = x)
    trace = pm.sample(samples=niter, nobjs = 4)


# In[62]:


# traceplot
pm.traceplot(trace)
pass


# In[63]:


# extract values
mu_samps_pymc = trace.get_values('mu', chains = [0,1,2,3])
fig, ax = plt.subplots()
ax.plot(mu_samps_pymc)
pass


# In[64]:


# summary
pm.summary(trace).round(2)


# In[65]:


# diagnostics: Gelman-Rubin
print(pm.diagnostics.gelman_rubin(trace))


# In[66]:


# diagnostics: n effective
print(pm.diagnostics.effective_n(trace))


# ## Linear regression
# 
# Likelihood:
# 
# $$ y \sim N(ax+b, \sigma^2)$$
# 
# Priors:
# 
# $$ a \sim N(0,100) $$
# $$ b \sim N(0,100) $$
# $$ a \sim U(0,20) $$
# 

# ### Data generation

# In[ ]:


##############################################
# linear regression
##############################################


# In[67]:


n = 11
a_true = 6
b_true = 2
x = np.linspace(0, 1, n)
y = a_true*x + b_true + np.random.randn(n)

stan_data = {
             'n': n,
             'x': x,
             'y': y
            }


# In[75]:


from pymc3.distributions import InverseGamma, Normal


# In[81]:


with pm.Model() as model1:
    # Define priors
    s = InverseGamma('s', 2, 3)
    a = Normal('a', 0, sd=10)
    b = Normal('b', 0, sd=10)

    likelihood = Normal('y', mu=b + a * x, sd=np.sqrt(s), observed=y)

    trace = pm.sample(samples=niter, nobjs = 4) 


# In[82]:


# traceplot
pm.traceplot(trace)
pass


# ## Binomial likelihood

# In[ ]:


##############################################
# binomial likelihood
##############################################


# In[118]:


def invlogit(x):
    return(np.exp(x) / (1 +  np.exp(x)))

n = 1000
x = np.random.randn(n)
alpha_true = -0.3
beta_true = 0.7
ps = alpha_true + beta_true*x 
ps = invlogit(ps)
y = [np.random.binomial(10, p) for p in ps]
plt.hist(ps)
pass


# In[119]:


import theano.tensor as t

def invlogit(x):
    return t.exp(x) / (1 + t.exp(x))


# In[129]:


niter = 10000
with pm.Model() as model:
    # define priors
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta = pm.Normal('beta', mu=0, sd=1)

    # define likelihood
    p = invlogit(alpha + beta*x)
    y_obs = pm.Binomial('y_obs', n=10, p=p, observed=y)

    # inference
    trace = pm.sample(samples=niter, nobjs = 4) 


# In[130]:


# traceplot
pm.traceplot(trace)
pass

