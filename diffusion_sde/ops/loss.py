import torch
from diffusion_sde.models.utils import get_score_fn

def get_sde_loss_fn(sde, device, reduce_mean=True, continuous=True, eps=1e-5):
    
    """Create a loss function for training with arbirary SDEs.
    
    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        eps: A `float` number. The smallest time step to sample from.
        
    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        
    def loss_score(model, img):
        """Evaluates the loss of a score function.

        Args:
            model: a score model
            img: a mini-batch of images
            cond: the conditioning

        Returns:
            evaluated loss
        """
        # print(img.shape)
        guide_img = img[:, [1], ...]
        disturb_img = img[:, [0], ...]

        t = torch.rand(disturb_img.shape[0], device=device) * (sde.T - eps) + eps
        z = torch.randn_like(disturb_img)

        score_fn = get_score_fn(sde, model, continuous=continuous)
        mean_disturb_img, std_disturb_img = sde.marginal_prob(disturb_img, t)
        perturbed_img = mean_disturb_img + std_disturb_img[:, None, None, None] * z
        perturbed_img = torch.cat([perturbed_img, guide_img], dim=1)
        score = score_fn(perturbed_img, t)
        losses = torch.square(score * std_disturb_img[:, None, None, None] + z)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        
        return torch.mean(losses)

    def loss_fn(model, batch):
        """Compute the loss function.
        
        Args:
            model: A score model.
            batch: A mini-batch of training data.
        
        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        
        # Load the x batch of dataset.
        x = batch
        x = x.to(device)
        
        # Calculate the losses. 
        loss = loss_score(model, x)
        
        return loss

    return loss_fn