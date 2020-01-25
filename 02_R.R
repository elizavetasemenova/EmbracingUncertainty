rm(list = ls())

########################################
# prioir x likelihood = posterior
########################################
successes = 6
tosses = 9

# define grid
grid_points = 100
p_grid <- seq( from=0 , to=1 , length.out=grid_points )

# compute likelihood at each value in grid
likelihood <- dbinom( successes , size=tosses , prob=p_grid)

computePosterior = function(likelihood, prior){
  # compute product of likelihood and prior
  unstd.posterior <- likelihood * prior
  
  # standardize the posterior, so it sums to 1
  posterior <- unstd.posterior / sum(unstd.posterior)
  
  par(mfrow=c(1,3))
  plot( p_grid , prior, type="l", main="Prioir")
  plot( p_grid , likelihood, type="l", main="Likelihood")
  plot( p_grid , posterior , type="l", main="Posterior")
  
}


prior1 <- rep( 1 , length(p_grid) )
computePosterior(likelihood, prior1)

prior2 <- ifelse( p_grid < 0.5 , 0 , 1 )
computePosterior(likelihood, prior2)

prior3 <- exp( -5*abs( p_grid - 0.5 ) )
computePosterior(likelihood, prior3)

##############################################
# the Monte Carlo method - compute pi
##############################################

in_circle <- function(x,y,r){
  sqrt(x^2 + y^2) <= r^2
}

approx_pi <- function(r, n){
  xs <- c()
  ys <- c()
  cols <- c()
  
  count = 0
  
  for (i in 1:n){
    x <- runif(1)
    y <- runif(1)
    xs <- c(xs, x)
    ys <- c(ys, y)
    
    if (in_circle(x, y, r)){
      count <- count + 1
      cols <- c(cols, "red")
    } else {
      cols <- c(cols, "steelblue")
    }
  }
  
  pi_appr <- round(4*count/n, digits = 3)
  
  plot(xs, ys, pch=19, col = cols, yaxt = 'n', xaxt='n')
  
}

r = 1
n = 100

for (n in 5 * 10^c(1,2,3)){
  approx_pi(r,n)
}

##############################################
# the Monte Carlo method - integration
##############################################

x <- seq(from=0 , to=1, length.out=100)
plot(x, exp(x), type='l')

pts <- matrix(runif(100 * 2), nrow=100, ncol = 2)
pts[,2] <- pts[,2] * exp(1)

cols <- rep("steelblue", 100)

for (i in 1:100){
  if (pts[i, 2] > exp(pts[i,1]))
    cols[i] <- "red"
}

plot(pts[,1], pts[,2], pch=19, col = cols, ylim = c(0, exp(1)))
lines(x, exp(x), type='l')

for (n in 10^c(1,2,3,4,5,6,7,8)){
  pts <- matrix(runif(n * 2), nrow=n, ncol = 2)
  pts[,2] <- pts[,2] * exp(1)
  count <- sum(pts[,2] < exp(pts[,1]))
  volume <- exp(1)
  sol <- (volume * count) / n
  print(sol)
}

##############################################
# coin tossing
##############################################
n = 4
h = 3
p = h/n

a = 10
b = 10


beta_binomial <- function(n,h,a,b){
  p <- h/n
  mu <- mean(rbinom(100, n, p))
  
  thetas <- seq(0,1, length.out = 200)
  prior <- dbeta(thetas, a, b)
  
  post <- dbeta(thetas, h+a, n-h+b)
  
  likelihood <- rep(NA, length(thetas))
  
  for (i in 1:length(thetas)){
    p <- thetas[i]
    likelihood[i] <-  n * dbinom(h, n , p)
  }
  
  plot(thetas, prior, type='l', col = "blue", ylim = c(0, max(prior, post, likelihood)))
  lines(thetas, post, col = "red")
  lines(thetas, likelihood, col = "green")
  abline(v = (h+a-1)/(n+a+b-2), col = "red")
  abline(v = mu / n, col = "green")
}


beta_binomial(100, 80, 10, 10)

beta_binomial(4, 3, 10, 10)

beta_binomial(4, 3, 2, 2)

beta_binomial(4, 3, 1, 1)

##############################################
# Metropolis-Hastings
##############################################

target <- function(likelihood, prior, n, h, theta){
  if (theta < 0 | theta > 1){
    return(0)
  } else {
    
  }
}


n = 100
h = 61
a = 10
b = 10

sigma = 0.3

naccept = 0
theta = 0.1
niters = 10000

samples <- rep(0, niters +1)
samples[1] <- theta

for (i in 1:niters){
  theta_p <- theta + rnorm(1, 0, sigma)
  
  target <- ifelse(theta < 0 | theta > 1, 0 , dbinom(h, n , theta) * dbeta(theta, a, b))
    
  taregt_p <- ifelse(theta_p < 0 | theta_p > 1, 0 , dbinom(h, n , theta_p) *  dbeta(theta_p, a, b))
  rho <- min(1, taregt_p / target)
  u <- runif(1)
  
  if (u < rho){
    naccept <- naccept + 1
    theta <- theta_p
  }
  
  samples[i+1] <- theta
  
}

print(paste0("Portion of accepted steps = ", naccept/niters))

nmcmc <- round(length(samples)/2)

library(scales)
s <- samples[nmcmc:length(samples)]
pr <- rbeta(nmcmc, a, b)
thetas <- seq(0, 1, length.out = 200)
ps <- 100 * dbeta(thetas, h+a, n-h+b)

hist(s, col = alpha("blue", 0.2), xlim = c(0, 1), ylim = c(0, 1000))
hist(pr, add = TRUE, col = alpha("red", 0.2))
lines(thetas, ps, col = "red")

mh_coin <- function(niters, n, h, theta, sigma){
  
  samples <- rep(0, niters +1)
  samples[1] <- theta
  
  for (i in 1:niters){
    theta_p <- theta + rnorm(1, 0, sigma)
    
    target <- ifelse(theta < 0 | theta > 1, 0 , dbinom(h, n , theta) * dbeta(theta, a, b))
    
    taregt_p <- ifelse(theta_p < 0 | theta_p > 1, 0 , dbinom(h, n , theta_p) *  dbeta(theta_p, a, b))
    rho <- min(1, taregt_p / target)
    u <- runif(1)
    
    if (u < rho){
      naccept <- naccept + 1
      theta <- theta_p
    }
    
    samples[i+1] <- theta
    
  }
  
  return(samples)
  
}

n = 100
h = 61
sigma = 0.05
niters = 100

theta_range <- seq(0.1, 1, by = 0.2)

for (theta in theta_range){
  if (theta == theta_range[1]){
    chains <- mh_coin(niters, n, h, theta, sigma)
  } else {
    chains <- rbind(chains, mh_coin(niters, n, h, theta, sigma))
  }
}

cols <- c("red", "blue", "darkgreen", "magenta", "coral")
plot(1, type="n", xlab="", ylab="", xlim=c(0, niters+1), ylim=c(min(chains), max(chains)))
for (i in 1:nrow(chains)){
  lines(chains[i,], col = cols[i])
}

########################################
# Stan
########################################
library(rstan)

stan_data <- list(
  n = 100,
  h = 61
)

# set directiry to current
setwd("~/Box Sync/32_AMLD")

fit <- stan(
  file = "02_coin.stan",  # Stan program
  data = stan_data,    # named list of data
  chains = 4,             # number of Markov chains
  warmup = 5000,          # number of warmup iterations per chain
  iter = 10000,            # total number of iterations per chain
  cores = 2,              # number of cores (could use one per chain)
  refresh = 0             # no progress shown
)

print(fit)
traceplot(fit)
density_plot(fit)
samples(fit)

p <- extract(fit, pars = c("p"))$p
hist(p)

##############################################
# normal distribution
##############################################

N <- 2000
y <- rnorm(N)
hist(y)
stan_data <- list(
  N = N,
  obs = y
)

fit <- stan(
  file = "02_norm_mu.stan",  # Stan program
  data = stan_data,    # named list of data
  chains = 4,             # number of Markov chains
  warmup = 5000,          # number of warmup iterations per chain
  iter = 10000,            # total number of iterations per chain
  cores = 2,              # number of cores (could use one per chain)
  refresh = 0             # no progress shown
)

print(fit)
traceplot(fit)

fit <- stan(
  file = "02_norm_mu_sigma.stan",  # Stan program
  data = stan_data,    # named list of data
  chains = 4,             # number of Markov chains
  warmup = 5000,          # number of warmup iterations per chain
  iter = 10000,            # total number of iterations per chain
  cores = 2,              # number of cores (could use one per chain)
  refresh = 0             # no progress shown
)

print(fit)
traceplot(fit)

##############################################
# linear regression
##############################################
n = 100
a_true = 6
b_true = 2
x = seq(0, 1, length.out = n)
y = a_true*x + b_true + rnorm(n)

plot(x, y, ylab= "", xlab= "", pch= 19, col="coral")
lines(x, a_true*x + b_true)

stan_data <- list(
  n = n,
  x = x,
  y = y
)

fit <- stan(
  file = "02_lin_reg.stan",  # Stan program
  data = stan_data,    # named list of data
  chains = 4,             # number of Markov chains
  warmup = 5000,          # number of warmup iterations per chain
  iter = 10000,            # total number of iterations per chain
  cores = 2,              # number of cores (could use one per chain)
  refresh = 0             # no progress shown
)

print(fit)
traceplot(fit, pars = c("a", "b", "sigma"))

##############################################
# binomial likelihood
##############################################
invlogit <- function(x){
  exp(x) /(1+exp(x))
}

n <- 10000
x <- rnorm(n)
alpha_true <- -0.3
beta_true <- 0.7
ps <- alpha_true + beta_true*x 
ps <- invlogit(ps)
y <- rep(NA, length(ps))
for (i in 1:length(ps)){
  p <- ps[i]
  y[i] <- rbinom(1, 10, p)
}
hist(ps,  ylab= "", xlab= "")

stan_data <- list(
  n = n,
  x = x,
  y = y
)

fit <- stan(
  file = "02_binom.stan",  # Stan program
  data = stan_data,    # named list of data
  chains = 4,             # number of Markov chains
  warmup = 5000,          # number of warmup iterations per chain
  iter = 10000,            # total number of iterations per chain
  cores = 2,              # number of cores (could use one per chain)
  refresh = 0             # no progress shown
)

print(fit)
traceplot(fit, pars = c("alpha", "beta"))