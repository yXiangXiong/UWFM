import torch

def residual_distribution(out_mean, out_1alpha, out_beta, target, mask):
    # residual distribution loss based on generalized normal distribution
    alpha_eps, beta_eps = 1e-5, 1e-1
    out_1alpha += alpha_eps
    out_beta += beta_eps
    
    factor = out_1alpha
    resi = torch.abs(out_mean - target)
    resi = (resi*factor*out_beta).clamp(min=1e-6, max=50)
    if torch.sum(resi != resi) > 0:
        print('resi has nans!!')
        return None

    log_1alpha = torch.log(out_1alpha)
    log_beta = torch.log(out_beta)
    lgamma_beta = torch.lgamma(torch.pow(out_beta, -1))
    
    if torch.sum(log_1alpha != log_1alpha) > 0:
        print('log_1alpha has nan')
        print(lgamma_beta.min(), lgamma_beta.max(), log_beta.min(), log_beta.max())
    if torch.sum(lgamma_beta != lgamma_beta) > 0:
        print('lgamma_beta has nan')
    if torch.sum(log_beta != log_beta) > 0:
        print('log_beta has nan')

    loss = torch.mean((resi - log_1alpha + lgamma_beta - log_beta) * mask)

    return loss