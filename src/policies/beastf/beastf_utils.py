import torch

def create_bidirectional_mask(batch_size, seq_length, device):
    """
    In a bidirectional mask, every token can attend to every other token,
    allowing full visibility in both directions.
    
    Args:
        batch_size (int): Batch size
        seq_length (int): Sequence length (both target and source length for self-attention)
        device: Device to create tensor on
        
    Returns:
        torch.FloatTensor: Bidirectional mask with shape (batch_size, 1, seq_length, seq_length)
    """
    # For bidirectional attention, we want all positions to be visible
    # This means the mask should be all zeros (allowing attention everywhere)
    
    # Create a tensor with shape (batch_size, 1, seq_length, seq_length) filled with zeros
    # In attention masks, 0.0 means "attend to this position"
    bidirectional_mask = torch.zeros((batch_size, 1, seq_length, seq_length), device=device)
    
    return bidirectional_mask


def token_prediction_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes token-level prediction accuracy for a batch.

    Args:
        logits (torch.Tensor): Model output logits of shape (batch_size, num_classes, seq_len, ...)
        targets (torch.Tensor): Ground-truth class indices of shape (batch_size, seq_len, ...)

    Returns:
        float: Accuracy as a percentage (0-100).
    """

    # Compute accuracy
    correct = (preds == targets).sum().item()
    total = targets.numel()  # Total number of tokens

    return 100.0 * correct / total if total > 0 else 0.0  # Return accuracy as percentage
    