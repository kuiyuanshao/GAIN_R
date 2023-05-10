library(caret)
library(torch)
library(progress)

gain <- function(data, device = "cpu", batch_size = 128, hint_rate = 0.9, alpha = 100, n = 10000){
  
  nRow <- dim(data)[1]
  nCol <- dim(data)[2]
  
  H_dim <- nCol
  
  norm_result <- normalize(data)
  norm_data <- norm_result$norm_data
  norm_parameters <- norm_result$norm_parameters
  
  norm_data[is.na(norm_data)] <- 0
  
  data_mask <- 1 - is.na(data)
  norm_data <- as.matrix(norm_data)
  
  norm_data <- torch::torch_tensor(norm_data)$to(device = device)
  data_mask <- torch::torch_tensor(data_mask)$to(device = device)
  
  norm_data <- torch_data(norm_data)
  data_mask <- torch_data(data_mask)
  
  GAIN_Generator <- torch::nn_module(
    initialize = function(nCol, H_dim){
      self$seq <- torch::nn_sequential()
      self$seq$add_module(module = torch::nn_linear(nCol * 2, H_dim),
                          name = "Linear1")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation1")
      self$seq$add_module(module = torch::nn_linear(H_dim, H_dim),
                          name = "Linear2")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation2")
      self$seq$add_module(module = torch::nn_linear(H_dim, nCol),
                          name = "Linear3")
      self$seq$add_module(module = torch::nn_sigmoid(),
                          name = "Output")
    },
    forward = function(input){
      input <- self$seq(input)
      input
    }
  )
  
  GAIN_Discriminator <- torch::nn_module(
    initialize = function(nCol, H_dim){
      self$seq <- torch::nn_sequential()
      self$seq$add_module(module = torch::nn_linear(nCol * 2, H_dim),
                          name = "Linear1")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation1")
      self$seq$add_module(module = torch::nn_linear(H_dim, H_dim),
                          name = "Linear2")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation2")
      self$seq$add_module(module = torch::nn_linear(H_dim, nCol),
                          name = "Linear3")
      self$seq$add_module(module = torch::nn_sigmoid(),
                          name = "Output")
    },
    forward = function(input){
      input <- self$seq(input)
      input
    }
  )
  
  G_layer <- GAIN_Generator(nCol, H_dim)$to(device = device)
  D_layer <- GAIN_Discriminator(nCol, H_dim)$to(device = device)
  
  G_solver <- torch::optim_adam(G_layer$parameters)
  D_solver <- torch::optim_adam(D_layer$parameters)
  
  generator <- function(X, M){
    input <- torch_cat(list(X, M), dim = 2)
    return (G_layer(input))
  }
  discriminator <- function(X, H){
    input <- torch_cat(list(X, H), dim = 2)
    return (D_layer(input))
  }
  
  G_loss <- function(X, M, H){
    G_sample <- generator(X, M)
    X_hat <- X * M + G_sample * (1 - M)
    D_prob <- discriminator(X_hat, H)
    
    G_loss1 <- -torch_mean((1 - M) * torch_log(D_prob + 1e-8))
    mse_loss <- torch_mean((M * X - M * G_sample) ^ 2) / torch_mean(M)
    
    return (G_loss1 + alpha * mse_loss)
  }
  D_loss <- function(X, M, H){
    G_sample <- generator(X, M)
    X_hat <- X * M + G_sample * (1 - M)
    D_prob <- discriminator(X_hat, H)
    D_loss1 <- -torch_mean(M * torch_log(D_prob + 1e-8) + (1 - M) * torch_log(1 - D_prob + 1e-8))
    return (D_loss1)
  }
  
  
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta",
    clear = FALSE, total = n, width = 60)
  
  for (i in 1:n){
    ind_batch <- new_batch(norm_data, data_mask, batch_size, device)
    X_mb <- ind_batch[[1]]
    M_mb <- ind_batch[[2]]
    
    Z_mb <- ((-0.01) * torch::torch_rand(c(batch_size, nCol)) + 0.01)$to(device = device)
    H_mb <- 1 * (((-1) * torch::torch_rand(c(batch_size, nCol)) + 1)$to(device = device) < hint_rate)

    H_mb <- M_mb * H_mb
    X_mb <- M_mb * X_mb + (1 - M_mb) * (Z_mb)
    H_mb$to(device = device)
    X_mb$to(device = device)
    
    d_loss <- D_loss(X_mb, M_mb, H_mb)
    
    D_solver$zero_grad()
    
    d_loss$backward()
    
    D_solver$step()
    
    g_loss <- G_loss(X_mb, M_mb, H_mb)
    
    G_solver$zero_grad()
    
    g_loss$backward()
    
    G_solver$step()
    
    pb$tick(tokens = list(what = "GAIN   "))
    Sys.sleep(2 / 100)
  }
  
  Z <- ((-0.01) * torch::torch_rand(c(nRow, nCol)) + 0.01)$to(device = device)
  X <- data_mask$torch.data * norm_data$torch.data + (1 - data_mask$torch.data) * Z
  X$to(device = device)
  
  G_sample <- generator(X, data_mask$torch.data)
  
  imputed_data <- data_mask$torch.data * X + (1 - data_mask$torch.data) * G_sample
  
  imputed_data <- imputed_data$detach()$cpu()
  
  imputed_data <- renormalize(imputed_data, norm_parameters)
  
  imputed_data <- data.frame(as.matrix(imputed_data))
  
  names(imputed_data) <- names(data)
  
  G_sample <- renormalize(G_sample, norm_parameters)
  
  G_sample <- data.frame(as.matrix(G_sample))
  
  names(G_sample) <- names(data)
  
  return (list(imputed_data, G_sample))
}