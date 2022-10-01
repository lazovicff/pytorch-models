import torch
import torch.nn as nn

print(torch.backends.mps.is_available())

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

learning_rate = 0.1
n_iters = 100

n_samples, n_features = x.shape
# n_samples = 4, n_features = 1
print(f'n_samples: {n_samples}, n_features: {n_features}')
# input and output size is the same
model = nn.Linear(n_features, n_features)
# define the loss function
loss_function = nn.MSELoss()
# define the optimizer function, the one that updates the weights
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Start the training loop
for epoch in range(n_iters):
	# Forward propagation
	y_pred = model(x)
	# Calculate the loss
	loss = loss_function(y, y_pred)
	# Calculate the gradient
	loss.backward()
	# Update the weights
	optimizer.step()
	# clear the gradients
	optimizer.zero_grad()
	# Print out the loss
	[w, b] = model.parameters()
	print(f'epoch {epoch}: w = {w}, loss = {loss:.8f}')

x_test = torch.tensor([5], dtype=torch.float32)
print(f'Prediction after training: f(5) = {model(x_test)}')