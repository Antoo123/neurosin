import torch
import numpy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize']=(30.0,7.0)

x_train=torch.rand(1000 )
x_train=x_train*50.0-25.0

y_sub_train=torch.cos(x_train)**2

noise=torch.randn(y_sub_train.shape)/10.


y_train=y_sub_train+noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_val=torch.linspace(-25,25,1000)
y_val=torch.cos(x_val.data)**2
x_val.unsqueeze_(1)
y_val.unsqueeze_(1)
# plt.plot(x_val.numpy(),y_val.numpy(),'o')
# plt.show()
# plt.plot(x_train.numpy(),y_train.numpy(),'o')
# plt.show()
class Ournet(torch.nn.Module):
	def __init__(self,n_hid_n):
		super(Ournet,self).__init__()
		self.fc1=torch.nn.Linear(1,n_hid_n)
		self.act1=torch.nn.Sigmoid()
		self.fc2=torch.nn.Linear(n_hid_n,n_hid_n)
		self.act2=torch.nn.Sigmoid()
		self.fc3=torch.nn.Linear(n_hid_n,1)
	def forward(self,x):
		x=self.fc1(x)
		x=self.act1(x)
		x=self.fc2(x)
		x=self.act2(x)
		x=self.fc3(x)
		# x=self.act3(x)
		return x 
our_net=Ournet(11)
def predict(net,x,y):
	y_pred=net.forward(x)
	plt.plot(x.numpy(),y.numpy(),'o',label='123')
	plt.plot(x.numpy(),y_pred.data.numpy(),'o',c='r')
	plt.show()
# predict(our_net,x_val,y_val) 
optimizer=torch.optim.Adam(our_net.parameters(),lr=0.001)
def loss(pred,true):
	sq= (pred-true)**2
	return sq.mean()
for e in range(10000):
	optimizer.zero_grad()
	y_pred=our_net.forward(x_train)
	loss_val=loss(y_pred,y_train)
	print(loss_val)
	loss_val.backward()
	optimizer.step()
predict(our_net,x_val,y_val)