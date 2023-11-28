import torch 

def get_score_fn(sde, model, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        model: A score model.

    Returns:
        A score function.
    """

    def score_fn(x, t):
        """Evaluate the score of a given batch of images:
        
        Args:
            x: A batch of images.
            t: time t sampled from a uniform distribution.
        
        Returns:
            The evaluated score.
        
        """
        if continuous:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
            labels = t * 999
            score = model(x, labels)
            std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
            labels = t * (sde.N - 1)
            score = model(x, labels)
            std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

        score = -score / std[:, None, None, None]
        return score

    return score_fn