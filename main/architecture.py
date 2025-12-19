"""
Simple MLP for Restoring Force prediction.
"""

import torch
import torch.nn as nn


class RFMLP(nn.Module):
    """
    Multi-Layer Perceptron for predicting Restoring Force from (x_rel, v_rel).
    
    Architecture:
        Input (2) -> Hidden1 (8) -> Hidden2 (4) -> Output (1)
    """
    
    def __init__(self):
        super(RFMLP, self).__init__()
        
        # Layer 1: 2 inputs -> 8 neurons
        self.fc1 = nn.Linear(2, 8)
        
        # Layer 2: 8 -> 4 neurons
        self.fc2 = nn.Linear(8, 4)
        
        # Output: 4 -> 1 neuron
        self.fc3 = nn.Linear(4, 1)
        
        # Activation function
        self.activation = nn.Tanh()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 2) containing [x_rel, v_rel]
        
        Returns:
            RF predictions of shape (batch_size, 1)
        """
        # Layer 1
        x = self.fc1(x)
        x = self.activation(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.activation(x)
        
        # Output (no activation - linear)
        x = self.fc3(x)
        
        return x


def count_parameters(model):
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = RFMLP()
    print(f"Model architecture:\n{model}")
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 1024
    x_test = torch.randn(batch_size, 2)
    output = model(x_test)
    print(f"\nInput shape: {x_test.shape}")
    print(f"Output shape: {output.shape}")
