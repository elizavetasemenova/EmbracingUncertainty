data{
 int<lower=0> N;
 real obs[N];
}

parameters{
 real mu;
 real<lower=0> s;
}

model{
 mu ~ normal(0, 0.5);
 s ~ inv_gamma(2, 3);
 obs ~ normal(mu, s);
}

