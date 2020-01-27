using Distributions
using Plots
using StatsBase
using Turing

##############################################
# prioir x likelihood = posterior
##############################################

success=6

tosses=9

# Create a distribution with n = 9 (e.g. tosses) and p = 0.5.

d = Binomial(tosses, 0.5)
pdf(d, success)


# define grid
grid_points = 100
p_grid =  range(0, stop = 1, length = grid_points)

# compute likelihood at each point in the grid
likelihood = [pdf(Binomial(tosses, p), success) for p in p_grid]

# define prior
prior = ones(length(p_grid));

# As Uniform prior has been used, unstandardized posterior is equal to likelihood

# compute product of likelihood and prior
posterior = likelihood .* prior;

function computePosterior(likelihood, prior)
   
    # compute product of likelihood and prior
    unstd_posterior = likelihood .* prior

    # standardize posterior
    posterior = unstd_posterior / sum(unstd_posterior)
    
    p1 = plot(p_grid, prior, title = "Prior")
    p2 = plot(p_grid, likelihood , title = "Likelihood")
    p3 = plot(p_grid, posterior, title = "Posterior")
    
    plot(p1, p2, p3, layout=(1, 3), label="")

end

prior1 = ones(length(p_grid))
posterior1 = computePosterior(likelihood, prior1)

#prior2 = 2 * (p_grid .>= 0.5)
prior2 = 0.5 * (p_grid .>= 0.5)
posterior2 = computePosterior(likelihood, prior2)

prior3 = exp.(-5 * abs.(p_grid .- 0.5))
posterior3 = computePosterior(likelihood, prior3)

##############################################
# the Monte Carlo method - compute pi
##############################################

function in_circle(x, y, r)
    sqrt(x^2 + y^2) <= r
end

function approx_pi(r, n)
    
    xs, ys, cols = [], [], []
    
    count = 0

    for i in range(1, step=1, stop=n)
        x = rand(Uniform(0,1))
        y = rand(Uniform(0,1))
        append!(xs, x)
        append!(ys, y)

        if in_circle(x, y, r)
            count += 1
            cols = vcat(cols, :red)
        else
            cols = vcat(cols, :steelblue)
        end
    end

    pi_appr = round(4 * count/n, digits = 3)
    
    pl = scatter(xs, 
        ys, 
        color=cols, 
        size=(200,200),
        legend = false,
        xticks = false,
        yticks = false,
        framestyle = :box,
        title = "pi (approximately) = " * string(pi_appr),
        titlefontsize=font(7, "Calibri"))
    
    display(pl)
    
end

r = 1
n = 100

for n in 5 * 10 .^[1, 2, 3]
    approx_pi(r, n)
end

##############################################
# the Monte Carlo method - integration
##############################################

exp(1) - exp(0)

x = range(0, stop = 1, length = 100)
plot(x, exp.(x), size= [200,200], legend= false)

pts =  rand(Uniform(0,1), (100, 2)) # sample uniformly in the square
pts[:, 2] *= exp(1)

cols = fill(:steelblue, 100)

for i in range(1, step=1, stop=100)
    if pts[i,2] > exp(pts[i,1])     # acceptance / rejection step
        cols[i] = :red
    end
end

scatter!(pts[:, 1], pts[:, 2], color = cols, size=[250, 250], legend = false, xlim = [0,1], ylim = [0, exp(1)])

# Monte Carlo approximation

for n in 10 .^[1, 2, 3, 4, 5, 6, 7, 8]
    pts =  rand(Uniform(0,1), (n, 2))
    pts[:, 2] *= exp(1)
    count = sum(pts[:, 2] .< exp.(pts[:, 1]))
    volume = exp(1) * 1 # volume of region
    sol = (volume * count)/n    
    println(sol)
end

##############################################
# coin tossing
##############################################

n = 4
h = 3
p = h/n

a, b = 10, 10                   # hyperparameters
prior = Beta(a, b)              # prior
post = Beta(h+a, n-h+b)         # posterior

function beta_binomial(n, h, a, b)
    # frequentist
    p = h/n
    mu = mean(Binomial(n, p))
    
    # Bayesian
    thetas = range(0, stop=1, length=200)
    prior = pdf.(Beta(a, b), thetas)

    post = pdf.(Beta(h+a, n-h+b), thetas)
    
    likelihood = n * [pdf(Binomial(n, p), h) for p in thetas];
    plot(thetas, 
         prior, 
         size= [400, 400], 
         label = "Prior",
         color = :blue,
         xlim = [0, 1],
         xlabel = "theta",
         ylabel = "Density")
    plot!(thetas, post, label = "Posterior", color = :red)
    plot!(thetas, likelihood, label="Likelihood", color = :green, legend = :topleft)
    vline!([(h+a-1)/(n+a+b-2)], color = :red, linestyle = :dash, label="MAP")
    vline!([mu / n], color = :green, linestyle = :dash, label="MLE")
 
end

beta_binomial(100, 80, 10, 10)

beta_binomial(4, 3, 10, 10)

beta_binomial(4, 3, 2, 2)

beta_binomial(4, 3, 1, 1)

##############################################
# Metropolis-Hastings
##############################################

function target(likelihood, prior, n, h, theta)
    if (theta < 0 ||  theta > 1)
        return 0
    else
        return (pdf(likelihood(n, theta), h) * pdf(prior, theta))
    end
end

n = 100
h = 61
a = 10
b = 10
likelihood = Binomial
prior = Beta(a, b)
sigma = 0.3

naccept = 0
theta = 0.1
niters = 10000

samples = zeros(niters+1)
samples[1] = theta

for i=1:niters
    theta_p = theta + rand(Normal(0, sigma))
    rho = min(1, target(likelihood, prior, n, h, theta_p)/target(likelihood, prior, n, h, theta ))
    u = rand(Uniform(0,1))
    if u < rho
            naccept += 1
            theta = theta_p
    end
    samples[i+1] = theta
end


println("Portion of accepted steps = " * string(naccept/niters))

nmcmc = Int(round(length(samples)/2))

post = Beta(h+a, n-h+b)
thetas = range(0, stop=1, length=200)

histogram(samples[nmcmc:length(samples)] ,
          size = [500, 300],
    
          label="Distribution of posterior samples", alpha = 0.5,
          legend = :topleft)
histogram!(rand(prior, nmcmc), 
          label = "Distribution of prior samples", alpha = 0.5)
plot!(thetas, 50*[pdf(post, theta) for theta in thetas], color = :red, label = "True posterior")

function mh_coin(niters, n, h, theta, likelihood, prior, sigma)
    
    samples = [theta]
    while length(samples) < niters
        theta_p = theta + rand(Normal(0, sigma))
        rho = min(1, target(likelihood, prior, n, h, theta_p)/target(likelihood, prior, n, h, theta ))
        u = rand(Uniform(0,1))
        if u < rho
            theta = theta_p
        end
        append!(samples, theta)
    end
    
    return samples

end

n = 100
h = 61
lik = Binomial
prior = Beta(a, b)
sigma = 0.05
niters = 100

chains = [mh_coin(niters, n, h, theta, lik, prior, sigma) for theta in range(0.1, stop=1, step=0.2)];

p = plot(chains[1], size= [500, 500], legend =:false, xlim = [0, niters], ylim = [0, 1])
for i in 2:length(chains)
    plot!(chains[i])
end
display(p)

##############################################
# Turing
##############################################

using Turing

@model mod(y) = begin
    # model definition
end

n = 100    # number of trials
h = 61     # number of successes

niter = 10000

@model coin(n, h) = begin
    
    # prior
    p ~ Beta(2, 2)
    
    # likelihood
    h ~ Binomial(n, p)
    
end

ch = sample(coin(n,h), NUTS(niter, 0.65));

show(ch)

# read samples into array
p = convert(Array{Float64}, ch[:p].value.data[:,:,1][:,1]);

histogram(p, size = [300, 300], legend = false, title = "posterior density")

# traceplot 
plot(p, size = [300, 300], legend = false, title = "traceplot")

function plot_par(par)
    p1 = histogram(par, size = [400, 300], legend = false, title = "posterior density")
    p2 = plot(par, title = "traceplot")
    plot(p1, p2, layout=(1, 2), label="")
end

plot_par(p)

##############################################
# hierarchical models
##############################################

@model coin_hier(n, h) = begin
    
    # hyperparameters    
    alpha_hyp ~ InverseGamma(10, 2)
    beta_hyp ~ InverseGamma(10, 2)
    
    # prior
    p ~ Beta(alpha_hyp, beta_hyp)
    
    # likelihood
    h ~ Binomial(n, p)
    
end

niter = 20000

ch = sample(coin_hier(n,h), NUTS(niter, 0.30));

show(ch)

# read samples into array
p = convert(Array{Float64}, ch[:p].value.data[:,:,1][:,1]);
plot_par(p)

alpha_hyp = convert(Array{Float64}, ch[:alpha_hyp].value.data[:,:,1][:,1]);
plot_par(alpha_hyp)

beta_hyp = convert(Array{Float64}, ch[:beta_hyp].value.data[:,:,1][:,1]);
plot_par(beta_hyp)

##############################################
# normal distribution
##############################################

N = 2000
y = rand(Normal(0,1), N)
histogram(y, size = [300, 300], legend = false)

@model norm_mu(y) = begin
    
    sigma = 1
    
    # prior
    mu ~ Normal(0,0.5)
    
    # likelihood    
    for i in eachindex(y)
        y[i] ~  Normal(mu, sigma)
   end
    
end

ch = sample(norm_mu(y), NUTS(niter, 0.65));

mu = ch[:mu].value.data[:,:,1]

plot_par(mu)

num_chains = 4
chains = mapreduce(c -> sample(norm_mu(y), NUTS(niter, 0.65)), chainscat, 1:num_chains)

mu = chains[:mu].value.data
plot(mu[:,:,1], label ="chain 1")
plot!(mu[:,:,2], label ="chain 2")
plot!(mu[:,:,3], label ="chain 3")
plot!(mu[:,:,4], label ="chain 4")

histogram(mu[:,:,1], alpha = 0.5, label = "chain 1")
histogram!(mu[:,:,2], alpha = 0.5, label = "chain 2")
histogram!(mu[:,:,3], alpha = 0.5, label = "chain 3")
histogram!(mu[:,:,4], alpha = 0.5, label = "chain 4")

@model norm_mu_sigma(y) = begin
        
    # priors
    mu ~ Normal(0,0.5)
    sigma ~ InverseGamma(2, 3)
    
    # likelihood    
    for i in eachindex(y)
        y[i] ~  Normal(mu, sigma)
   end
    
end

ch = sample(norm_mu_sigma(y), NUTS(niter, 0.65));

show(ch)

mu = ch[:mu].value.data[:,:,1]
sigma = ch[:sigma].value.data[:,:,1]
pl_mu = plot_par(mu)
vline!([0], color = :green, label="MLE")
pl_sigma = plot_par(sigma)
vline!([1], color = :green, label="MLE")
display(pl_mu)
display(pl_sigma)

##############################################
# linear regression
##############################################

n = 100
a_true = 6
b_true = 2
x = range(0, stop=1, length = n)
x = convert(Array, x)
y = a_true*x .+ b_true + rand(Normal(0,1), n);

plot(x, a_true*x .+ b_true, legend = false, size = [350, 350], color = :blue)
scatter!(x, y)

@model lin_reg(x, y) = begin
  
  a ~ Normal(0, 10)
  b ~ Normal(0, 10)
  lp = a * x .+ b
    
  s ~ InverseGamma(2, 3)
    
  for i in eachindex(y)
    y[i] ~ Normal(lp[i], sqrt(s))
  end
end

niter = 20000

ch = sample(lin_reg(x, y), NUTS(niter, 0.65));

a = ch[:a].value.data[:,:,1]
b = ch[:b].value.data[:,:,1]
s = ch[:s].value.data[:,:,1]
pl_a = plot_par(a)
vline!([a_true])
pl_b = plot_par(b)
vline!([b_true])
pl_s = plot_par(s)
vline!([1])
display(pl_a); 
display(pl_b)
display(pl_s)

##############################################
# binomial likelihood
##############################################

function invlogit(x)
  exp.(x) ./ (1 .+  exp.(x))
end

n = 1000
x = rand(Normal(), n)
alpha_true = -0.3
beta_true = 0.7
ps = alpha_true .+ beta_true*x 
ps = invlogit(ps)
y = [rand(Binomial(10, p)) for p in ps];
histogram(ps, bins = 20, title = "p", label = "", size = [300, 300])

@model binom(x, y) = begin
  
  alpha ~ Normal(0, 1)
  beta ~ Normal(0, 1)
  
  p = invlogit(alpha .+ beta*x)

  for i in eachindex(y)
    y[i] ~ Binomial(10, p[i])
  end
end

ch = sample(binom(x, y), NUTS(niter, 0.65));

show(ch)

alpha = ch[:alpha].value.data[:,:,1]
beta = ch[:beta].value.data[:,:,1]
pl_a = plot_par(alpha)
pl_b = plot_par(beta)
display(pl_a); 
display(pl_b); 

using Pkg
Pkg.status()
