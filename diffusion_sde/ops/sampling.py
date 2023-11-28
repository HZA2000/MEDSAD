import torch
from diffusion_sde.configs.config import CFGS
from diffusion_sde.models.utils import get_score_fn

def get_sampler(sde, shape, device, eps=1e-3):
    """Creates an Euler_Maruyama_sampler function.
    
    Args:
        sde: An VPSDE instance.
        shape: Expected shape of a single sample.
        device: PyTorch device.
        eps: For numerical stability.

    Returns:
        A sampling function that returns samples.
    
    """
    batch_size = shape[0]
    
    def Euler_Maruyama_sampler(model, num_steps=sde.N):
        """Generate samples from score-based models with the Euler-Maruyama solver.

        Args:
            score_fn: A PyTorch model that represents the time-dependent score-based model.
            cond: Batch of images on which to condition the scores.

        Returns:
            Samples.    
        """
        
        with torch.no_grad():
            score_fn = get_score_fn(sde, model, continuous=True)
            timesteps = torch.linspace(sde.T, eps, num_steps, device=device)
            init_x = sde.prior_sampling(shape).to(device)
            dt = timesteps[0] - timesteps[1]
            x_t = init_x

            for time_step in timesteps:      
                batch_time_step = torch.ones(batch_size, device=device) * time_step
                z = torch.randn_like(x_t) 
                drift, diffusion = sde.sde(x_t, batch_time_step)
                score = score_fn(x_t, batch_time_step)
                x_mean = x_t + (drift - diffusion[:, None, None, None]**2 * score) * (-dt)
                x_t = x_mean + torch.sqrt(dt) * diffusion[:, None, None, None] * z
                
        # Do not include any noise in the last sampling step.
        return x_mean
    
    return Euler_Maruyama_sampler


def get_guide_sampler(sde, shape, device, eps=1e-3):
    """Creates an Euler_Maruyama_sampler function.
    
    Args:
        sde: An VPSDE instance.
        shape: Expected shape of a single sample.
        device: PyTorch device.
        eps: For numerical stability.

    Returns:
        A sampling function that returns samples.
    
    """
    batch_size = shape[0]
    
    def Euler_Maruyama_sampler(model, guide_img, num_steps=sde.N):
        """Generate samples from score-based models with the Euler-Maruyama solver.

        Args:
            score_fn: A PyTorch model that represents the time-dependent score-based model.
            cond: Batch of images on which to condition the scores.

        Returns:
            Samples.    
        """
        
        with torch.no_grad():
            score_fn = get_score_fn(sde, model, continuous=True)
            timesteps = torch.linspace(sde.T, eps, num_steps, device=device)
            init_x = sde.prior_sampling(shape).to(device)
            dt = timesteps[0] - timesteps[1]
            x_t = init_x

            for time_step in timesteps:      
                batch_time_step = torch.ones(batch_size, device=device) * time_step
                z = torch.randn_like(x_t) 
                drift, diffusion = sde.sde(x_t, batch_time_step)
                # print(x_t.shape, guide_img.shape)
                score_input = torch.cat([x_t, guide_img], dim=1)
                score = score_fn(score_input, batch_time_step)
                x_mean = x_t + (drift - diffusion[:, None, None, None]**2 * score) * (-dt)
                x_t = x_mean + torch.sqrt(dt) * diffusion[:, None, None, None] * z
                
        # Do not include any noise in the last sampling step.
        return x_mean
    
    return Euler_Maruyama_sampler