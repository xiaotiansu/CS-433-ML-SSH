import torch

from online import FeatureQueue

dim = 3
bs = 2

q = FeatureQueue(dim=dim, length=10)

print(q.queue)

for i in range(1,8):
	feat = torch.ones(bs, dim) * i
	temp = q.get()
	print(temp)
	q.update(feat)
	# print(q.queue)