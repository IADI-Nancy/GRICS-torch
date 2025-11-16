# Espirit Maps Parameters - trying to get rid of magic numbers in my code
acs_size = 48  
kernel_size = 6

# motion artifacts parameters
alpha = 0.5  # draw back to original position 

# Conjugate Gradient Parameters
iterations_cg = 500
tolerance_cg = 1e-2
lambda_ = 1e-3

# Nonlinear Conjugate Gradient Parameters
beta_ = "FR"  # Fletcher-Reeves "FR", Polak-Ribiere "PR", Hestenes-Stiefel "HS", Dai-Yuan "DY"

# Adam Parameters
learning_rate = 1e-3
iterations_adam = 150
adam_lambda = 1e-3

# Getter Espirit Maps Parameters
def get_acs_size():
    return acs_size

def get_kernel_size():
    return kernel_size

# Getter motion artifacts Parameter
def get_alpha():
    return alpha

# Getter Conjugate Gradient Parameter
def get_iterations_cg():
    return iterations_cg

def get_tolerance_cg():
    return tolerance_cg

def get_lambda():
    return lambda_

# Getter/Setter Nonlinear Conjugate Gradient Parameter
def get_beta():
    return beta_

def set_beta(beta):
    betas = ["FR", "PR", "HS", "DY"]
    if beta_ in betas:
        beta_ = beta
    else:
        raise ValueError(f"Invalid beta: {beta_!r}. Must be one of {betas}")
    
# Getter Adam Parameter
def get_learning_rate():
    return learning_rate

def get_iterations_adam():
    return iterations_adam

def get_adam_lambda():
    return adam_lambda