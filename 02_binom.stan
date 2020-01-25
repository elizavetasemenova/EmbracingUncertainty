data {
    int<lower=0> n;
    real x[n];
    int y[n];
}
transformed data {}
parameters {
    real alpha;
    real beta;
}
transformed parameters {
    real ps[n];
    for (i in 1:n) {
        ps[i] =  inv_logit(alpha + beta * x[i]);
     }
}
model {
  alpha ~ normal(0,1);
  beta ~ normal(0,1);
    
  for (i in 1:n){
    y[i] ~ binomial(10, ps[i]);
  }
    
}
generated quantities {}

