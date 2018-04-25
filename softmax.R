

softmax_rhs <- function(x,y, global_scale=0.01, slab_scale=5, slab_df=4, intercept_scale=10,
                      control=list(adapt_delta=0.99), ...) {
  #
  # Fit softmax regression model with regularized horseshoe prior.
  #
  
  if (is.vector(x))
    x <- matrix(x)
  if (is.factor(y)) {
    k <- length(levels(y))
    y <- as.numeric(y) # values 1,...,K
  } else {
    y <- as.numeric(y)
  }
  n <- nrow(x)
  d <- ncol(x)
  
  if (is.null(control$adapt_delta))
    control$adapt_delta <- 0.99
  
  data <- list(n=n, d=d, k=k, x=x, y=y, scale_icept=intercept_scale, scale_global=global_scale,
               slab_scale=slab_scale, slab_df=slab_df)
  stanmodel <- stan_model('stan/softmax_rhs.stan')
  stanfit <- sampling(stanmodel, data=data, control=control, ...)
  return(stanfit)
}



softmax_pred <- function(stanfit, x) {
  
  #
  # Compute predictions for softmax regression for given test inputs x.
  #
  
  e <- extract(stanfit)
  fit <- list(beta=e$beta, beta0=e$beta0)
  
  n <- ifelse(is.null(dim(x)), 1, nrow(x))
  S <- dim(fit$beta)[[1]] # number of samples
  K <- dim(fit$beta)[[2]] # number of classes
  f <- array(dim=c(n,K,S))
  mu <- array(dim=c(n,K,S))
  for (k in 1:K) {
    temp <- sweep(x %*% t(fit$beta[,k,]), 2, fit$beta0[,k], '+')
    f[,k,] <- temp
  }
  for (s in 1:S)
    mu[,,s] <- exp(f[,,s,drop=F]) / rowSums(exp(f[,,s,drop=F]))
  mu <- apply(mu, c(1,2), mean) # average over parameter draws
  
  return(mu)
}

