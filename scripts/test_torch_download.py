import torch
print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
print(f'torch.tensor([1]).to(0): {torch.tensor([1]).to(0)}')

assert torch.cuda.is_available()
assert str(torch.tensor([1]).to(0).device) == 'cuda:0'

print('Looks good.')