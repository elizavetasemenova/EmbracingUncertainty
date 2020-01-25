data {
  int<lower=0> n; // number of tosses
  int<lower=0> h; // number of heads
  
}
transformed data {}
parameters {
  real<lower=0, upper=1> p;
}
transformed parameters {}
model {
  p ~ beta(2, 2);
  h ~ binomial(n, p);
}

generated quantities {}
