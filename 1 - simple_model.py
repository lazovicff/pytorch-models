import torch

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
	# Forward propagation
	y_pred = w * x
	# Calculate the loss
	loss = ((y_pred - y) ** 2).mean()
	# Calculate the gradient of loss with respect to w
	loss.backward()
	# Update the weights
	with torch.no_grad():
		w -= learning_rate * w.grad
	# Clear the gradient graph
	w.grad.zero_()
	# Print the result at each epoch
	print(f'epoch {epoch}: w = {w}, loss = {loss:.8f}')

print(f'Prediction after training: f(5) = {w * 5}')