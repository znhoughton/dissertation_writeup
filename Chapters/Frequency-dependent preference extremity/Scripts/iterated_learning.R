#three functions included here. First one is the main function. The other two are the iterated learning functions: iterated_learning which is a parallelized iterated learning model, and iterated_learning_not_parallelized which, unsurprisingly, is the original iterated learning model we created without parallelization.
#details about the functions can be found in the noisy_channel_sims.Rmd file
noisy_channel_learning_prod_noise = function(p_theta, N, prior_mu, nu, p_noise, prior_prob_noise) {
  
  
  alpha_1 = prior_mu * nu        #from the beta distribution
  alpha_2 = (1 - prior_mu) * nu  #from the beta distribution
  
  for (i in 1:N) {
    generated_word = rbinom(n = 1, size = 1, prob = p_theta) # 1 for AandB, 0 for BandA
    
    prior_noise = rbinom(n = 1, size = 1, prob = 1-prior_prob_noise) # 1 if speaker produces the correct utterance, 0 if not
    
    if (generated_word == prior_noise) { #if the speaker produces A and B and intends to produce A and B, or produces A and B but intends to produce B and A
      
      un_normalized_p_hat_alpha = (alpha_1 / (alpha_1 + alpha_2)) * (1 - p_noise)
      un_normalized_p_hat_nonalpha = (1 - (alpha_1 / (alpha_1 + alpha_2))) * p_noise
      
    }
    
    else { #if we hear B and A: it should be similar to above, but in this case, we actually did hear B and A, so we multiply it by p * noise, not 1 - p_noise
      un_normalized_p_hat_alpha = (alpha_1 / (alpha_1 + alpha_2)) * p_noise #we mutiply this by 0.05 because we actually did hear B and A
      un_normalized_p_hat_nonalpha = (1 - (alpha_1 / (alpha_1 + alpha_2))) *  (1 - p_noise) #multiply this by 1-p_noise because we heard correctly
    }
    
    p_hat_alpha = un_normalized_p_hat_alpha / (un_normalized_p_hat_alpha + un_normalized_p_hat_nonalpha)
    p_hat_nonalpha = 1 - (un_normalized_p_hat_alpha / (un_normalized_p_hat_alpha + un_normalized_p_hat_nonalpha))
    
    alpha_1 = alpha_1 + p_hat_alpha
    alpha_2 = alpha_2 + p_hat_nonalpha
  }
  
  return(alpha_1 / (alpha_1 + alpha_2))
  #rbinom(p = alpha_1 / (alpha_1 + alpha_2))
  
}

iterated_learning = function(n_gen, n_sims, p_theta, N, prior_mu, nu, p_noise, prior_prob_noise, last_gen_only = F) { #set last_gen_only to T if you wish to save memory and only care about the last generation.
  #mu_df = tibble(posterior_mu = numeric(), generation = numeric(), estimated_p_theta = numeric())
  
  sim_function = function() {
    p_new_theta = p_theta
    map_dfr(1:n_gen, ~{
      #if (. %% 100 == 0) {
      # print(sprintf('current generation is: %s', .))
      #}
      generation = .
      posterior_mu = noisy_channel_learning_prod_noise(p_theta = p_new_theta, N = N, prior_mu = prior_mu, nu = nu, p_noise = p_noise, prior_prob_noise = prior_prob_noise)
      
      if(last_gen_only == F) {
        result = tibble(posterior_mu = posterior_mu, generation = generation, estimated_p_theta = p_new_theta)
        p_new_theta <<- posterior_mu #need to use the operator <<- to update p_new_theta on a global scope
      
        return(result)
      }
      else{
        p_new_theta <<- posterior_mu #need to use the operator <<- to update p_new_theta on a global scope
        if(generation == n_gen) {
          result = tibble(posterior_mu = posterior_mu, generation = generation, estimated_p_theta = p_new_theta)
        }
      }
    })
  }
  
  mu_df <- future_map_dfr(1:n_sims, ~sim_function())
  
  return(mu_df)
}

iterated_learning_not_parallelized = function(n_gen, n_sims, p_theta, N, prior_mu, nu, p_noise, prior_prob_noise) { #this version is included to make sure that the parallelized version yields similar results to this one, since underlyingly they should be the same
  mu_df = data.frame(matrix(ncol = 3, nrow = 0))
  colnames(mu_df) = c('posterior_mu', 'generation', 'estimated p_theta')
  for (i in 1:n_sims) {
    #if(i %% 5 == 0) {
    #print(sprintf('current simulation number: %s', i))
    #print(sprintf('current frequency value: %s', N))
    #print(i)
    #print(N)
    
    #}
    
    p_new_theta = p_theta  
    
    for(i in 1:n_gen) {
      if(i %% 100 == 0) {
        #print(sprintf('current generation is: %s', i))
      }
      generation = i
      posterior_mu = noisy_channel_learning_prod_noise(p_theta = p_new_theta, N=N, prior_mu = prior_mu, nu=nu, p_noise=p_noise, prior_prob_noise=prior_prob_noise)
      mu_df[nrow(mu_df) + 1,] = c(posterior_mu, generation, p_new_theta)
      p_new_theta = posterior_mu
    }
  }
  
  return(mu_df)
  
}


###testing what happens if they hear all AandB then BandA

noisy_channel_learning_prod_noise_ordered = function(p_theta, N, prior_mu, nu, p_noise, prior_prob_noise) {
  
  
  alpha_1 = prior_mu * nu        #from the beta distribution
  alpha_2 = (1 - prior_mu) * nu  #from the beta distribution
  
  num_alpha = rbinom(n = 1, size = N, prob = p_theta)
  num_nonalpha = N - num_alpha
  
  for (i in 1:num_alpha) {
    
    prior_noise = rbinom(n = 1, size = 1, prob = 1-prior_prob_noise)
    
    if (prior_noise == 1) {
      un_normalized_p_hat_alpha = (alpha_1 / (alpha_1 + alpha_2)) * (1 - p_noise)
      un_normalized_p_hat_nonalpha = (1 - (alpha_1 / (alpha_1 + alpha_2))) * p_noise
      
    }
    
    else { #if we hear B and A: it should be similar to above, but in this case, we actually did hear B and A, so we multiply it by p * noise, not 1 - p_noise
      un_normalized_p_hat_alpha = (alpha_1 / (alpha_1 + alpha_2)) * p_noise #we mutiply this by 0.05 because we actually did hear B and A
      un_normalized_p_hat_nonalpha = (1 - (alpha_1 / (alpha_1 + alpha_2))) *  (1 - p_noise) #multiply this by 1-p_noise because we heard correctly
    }
    
    p_hat_alpha = un_normalized_p_hat_alpha / (un_normalized_p_hat_alpha + un_normalized_p_hat_nonalpha)
    p_hat_nonalpha = 1 - (un_normalized_p_hat_alpha / (un_normalized_p_hat_alpha + un_normalized_p_hat_nonalpha))
    
    alpha_1 = alpha_1 + p_hat_alpha
    alpha_2 = alpha_2 + p_hat_nonalpha
  }
  
  for (i in 1:num_nonalpha) { #if the speaker intends to produce BandA
    
    prior_noise = rbinom(n = 1, size = 1, prob = 1-prior_prob_noise) #1 if speaker produces BandA
    
    if (prior_noise == 0) { #if they produce AandB
      un_normalized_p_hat_alpha = (alpha_1 / (alpha_1 + alpha_2)) * (1 - p_noise)
      un_normalized_p_hat_nonalpha = (1 - (alpha_1 / (alpha_1 + alpha_2))) * p_noise
      
    }
    
    else { #if we hear B and A: it should be similar to above, but in this case, we actually did hear B and A, so we multiply it by p * noise, not 1 - p_noise
      un_normalized_p_hat_alpha = (alpha_1 / (alpha_1 + alpha_2)) * p_noise #we mutiply this by 0.05 because we actually did hear B and A
      un_normalized_p_hat_nonalpha = (1 - (alpha_1 / (alpha_1 + alpha_2))) *  (1 - p_noise) #multiply this by 1-p_noise because we heard correctly
    }
    
    p_hat_alpha = un_normalized_p_hat_alpha / (un_normalized_p_hat_alpha + un_normalized_p_hat_nonalpha)
    p_hat_nonalpha = 1 - (un_normalized_p_hat_alpha / (un_normalized_p_hat_alpha + un_normalized_p_hat_nonalpha))
    
    alpha_1 = alpha_1 + p_hat_alpha
    alpha_2 = alpha_2 + p_hat_nonalpha
  }
  
  return(alpha_1 / (alpha_1 + alpha_2))
  #rbinom(p = alpha_1 / (alpha_1 + alpha_2))
  
}


# 
# iterated_learning_ordered = function(n_gen, n_sims, p_theta, N, prior_mu, nu, p_noise, prior_prob_noise, last_gen_only = F) { #set last_gen_only to T if you wish to save memory and only care about the last generation.
#   #mu_df = tibble(posterior_mu = numeric(), generation = numeric(), estimated_p_theta = numeric())
#   
#   sim_function = function() {
#     p_new_theta = p_theta
#     map_dfr(1:n_gen, ~{
#       #if (. %% 100 == 0) {
#       # print(sprintf('current generation is: %s', .))
#       #}
#       generation = .
#       posterior_mu = noisy_channel_learning_prod_noise_ordered(p_theta = p_new_theta, N = N, prior_mu = prior_mu, nu = nu, p_noise = p_noise, prior_prob_noise = prior_prob_noise)
#       
#       if(last_gen_only == F) {
#         result = tibble(posterior_mu = posterior_mu, generation = generation, estimated_p_theta = p_new_theta)
#         p_new_theta <<- posterior_mu #need to use the operator <<- to update p_new_theta on a global scope
#         
#         return(result)
#       }
#       else{
#         p_new_theta <<- posterior_mu #need to use the operator <<- to update p_new_theta on a global scope
#         if(generation == n_gen) {
#           result = tibble(posterior_mu = posterior_mu, generation = generation, estimated_p_theta = p_new_theta)
#         }
#       }
#     })
#   }
#   
#   mu_df <- future_map_dfr(1:n_sims, ~sim_function())
#   
#   return(mu_df)
# }