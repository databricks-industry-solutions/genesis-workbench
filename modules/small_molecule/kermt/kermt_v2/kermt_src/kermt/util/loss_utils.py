"""
Utility functions for loss computation and gradient normalization.
"""

import torch


class GradientNormalizedLoss(torch.autograd.Function):
    """
    Custom autograd function that normalizes gradients without changing the loss value.
    
    This is useful for balancing gradients from different loss components that may have
    different scales (e.g., high-dimensional reconstruction loss vs. low-dimensional KL loss).
    """
    
    @staticmethod
    def forward(ctx, loss, c):
        """
        Forward pass: simply returns the loss value.
        
        Args:
            loss (torch.Tensor): The original loss value.
            c (float): Normalization factor for the gradient.
        
        Returns:
            torch.Tensor: The same loss value.
        """
        ctx.c = c  # Store scaling factor for backward
        return loss.clone()  # Ensure no modification to loss value itself
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: scales the gradient by c.
        
        Args:
            grad_output (torch.Tensor): The gradient of the loss.
        
        Returns:
            torch.Tensor: Scaled gradient.
            None: No gradient needed for c.
        """
        c = ctx.c
        return grad_output * c, None  # Scale gradient by c


def normalize_loss_gradient(
    loss: torch.Tensor,
    c: float,
    normalize_gradient: bool = False,
    normalize_loss: bool = False,
) -> torch.Tensor:
    """
    Normalize the gradient and/or loss value by a constant factor.
    
    This is useful for balancing gradients from loss terms with different dimensionalities.
    For example, reconstruction loss over a 100-dim space vs. KL loss over a 512-dim latent space.
    
    Args:
        loss (torch.Tensor): The loss tensor to normalize.
        c (float): Normalization coefficient (typically 1/num_dimensions).
        normalize_gradient (bool): If True, scale the gradient during backprop.
        normalize_loss (bool): If True, scale the loss value itself.
    
    Returns:
        torch.Tensor: The (potentially) normalized loss.
    
    Examples:
        >>> # Normalize gradient by dimensionality
        >>> loss = reconstruction_loss.sum(-1)  # [batch_size]
        >>> c = 1.0 / num_features
        >>> normalized = normalize_loss_gradient(loss, c, normalize_gradient=True)
        
        >>> # Normalize both loss value and gradient
        >>> normalized = normalize_loss_gradient(loss, c, normalize_gradient=True, normalize_loss=True)
    """
    enabled = normalize_gradient or normalize_loss
    if not enabled:
        return loss
    
    if normalize_loss:
        loss = loss * c  # Normalize the loss value
    
    if normalize_gradient:
        loss = GradientNormalizedLoss.apply(loss, c)  # Apply gradient normalization
    
    return loss

