import torch

# Create a tensor with a unique pattern in each batch
batch_size = 3
seq_length = 4
d_model = 2
tensor = torch.arange(batch_size * seq_length * d_model).reshape(batch_size, seq_length, d_model)
print('Initial:',tensor)

# Flatten the tensor
flattened_tensor = tensor.view(batch_size, -1)

# Check the values in the flattened tensor
print('view:',flattened_tensor)

