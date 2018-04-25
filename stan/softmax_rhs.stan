
data {
  int<lower=0> k;             // number of classes
  int<lower=0> n;				      // number of observations
  int<lower=0> d;             // number of predictors
  int y[n];
  matrix[n,d] x;
  real<lower=0> scale_icept;	// prior std for the intercept
  real<lower=0> scale_global;	// scale for the half-t prior for tau
  real<lower=0> slab_scale;   // for the regularized horseshoe
  real<lower=0> slab_df;      // for the regularized horseshoe
}

parameters {
  vector[k] beta0;
  matrix[k,d] z;
  matrix<lower=0>[k,d] lambda;
  real<lower=0> tau;
  real<lower=0> caux;
}

transformed parameters {
  
  real<lower=0> c;
  matrix[k,d] beta; // regression coefficients
  matrix[k,n] f;
  matrix<lower=0>[k,d] lambda_tilde;
  
  c = slab_scale * sqrt(caux);
  lambda_tilde = sqrt( c^2 * square(lambda) ./ (c^2 + tau^2* square(lambda)) );
  beta = z .* lambda_tilde*tau;
  f = append_col(beta0,beta) * append_col(rep_vector(1,n), x)';
}

model {
  
  to_vector(z) ~ normal(0,1);
  to_vector(lambda) ~ cauchy(0,1);
  tau ~ cauchy(0, scale_global);
  caux ~ inv_gamma(0.5*slab_df, 0.5*slab_df);
  beta0 ~ normal(0,scale_icept);
  
  for (i in 1:n)
    y[i] ~ categorical_logit(col(f,i));
}

generated quantities {
  // compute log-likelihoods for loo
  vector[n] loglik;
  for (i in 1:n)
    loglik[i] = categorical_logit_lpmf(y[i] | col(f,i));
}



