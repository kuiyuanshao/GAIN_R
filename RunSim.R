
source("gain_torch.R")
source("utils_torch.R")

sim <- function(){
  n  <- 10000
  beta <- c(1, 1, 1)
  e_U <- c(sqrt(4),sqrt(1))
  mx <- 0; sx <- 1; zrange <- 1; zprob <- .5
  
  simZ   <- rbinom(n, zrange, zprob)
  simX   <- (1-simZ)*rnorm(n, 0, 1) + simZ*rnorm(n, 0.5, 1)
  epsilon <- rnorm(n, 0, 1)
  simY    <- beta[1] + beta[2]*simX + beta[3]*simZ + epsilon
  simX_tilde <- simX + rnorm(n, 0, e_U[1]*(simZ==0) + e_U[2]*(simZ==1))
  data_full <- data.frame(X_tilde=simX_tilde, Y=simY, X=simX, Z=simZ)
  
  return (data_full)
}

library(torch)
library(progress)
library(caret)
library(missMethods)
library(mice)

data_full <- sim()
save(data_full, file = "data_full.RData")
for (i in seq(0.1, 0.8, by = 0.1)){
  data <- delete_MCAR(data_full, p = 0.2)
  data_m <- 1 - is.na(data)
  
  mice_list <- mice(data)
  mice_data <- complete(mice_list, "all")
  gain_data <- list()
  for (j in 1:5){
    gain_imp <- gain(data, n = 100)
    gain_data[[j]] <- gain_imp
  }
  save(data, data_m, mice_data, gain_data, file = paste0("sim", "_", i, ".RData"))
}




