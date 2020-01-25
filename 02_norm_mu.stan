data{
 int<lower=0> N;
 real obs[N];
}

parameters{
 real mu;
}

model{
 mu ~ normal(0, 0.5);
 obs ~ normal(mu, 1);
}

