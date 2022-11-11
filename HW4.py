%clear
import numpy as np
import matplotlib.pyplot as plt
W1 = np.array([[0.30, 0.55, 0.20], [0.45, 0.50, 0.35]])
W2 = np.array([[0.15, 0.40, 0.25]]).T
eta = 0.01
b1 = np.array([[0.60, 0.70, 0.85]])
b2 = np.array([0.05])
i = np.array([[0,0],[0,1],[1,0],[1,1]])
o = np.array([0,1,1,1])
Error = []
def z(W,i,b): #calculates net input function
    net = np.matmul(i,W)+b
    return net
def Sigmoid(z): # Calculates activation function output
	return (1/(1+np.exp(-z)))
def MSE(out, o): # Calculates mean squared error
    E= np.sum((out-o)**2)*0.25
    return E
def dSigmoid(z):
    return (Sigmoid(z))*(1-Sigmoid(z))
def dMSE(out, o):
    return np.sum(out-o)*0.5
def FF(i,W1,W2,b1,b2,o):
    z1 = z(W1,i,b1)
    a1= Sigmoid(z1)
    z2= z(W2,a1,b2)
    out=Sigmoid(z2)
    E= (MSE(out,o))
    return a1, out, z1,z2,E
a1, out, z1,z2,E = FF(i,W1,W2,b1,b2,o)

def BP(a1, out, z1,z2,E, o):
     d2 = np.multiply(dMSE(out,o),dSigmoid(z2))
     d1 = np.multiply(np.matmul(d2,W2.T),dSigmoid(z1))
     dW2 = np.dot(a1.T,d2)
     dW1 = np.dot(i.T,d1)
     return dW2, dW1
dW2, dW1 = BP(a1, out, z1,z2,E, o)

def GD(W1,W2,dW1,dW2):
    W1 = W1- eta*dW1
    W2 = W2- eta*dW2
    return W1, W2
W1, W2 = GD(W1,W2,dW1,dW2)
epochs = 10000
OUT = []
for n in range (epochs):
    a1, out, z1,z2,E = FF(i,W1,W2,b1,b2,o)
    #OUT.append[out]
    Error.append(E)
    dW2, dW1 = BP(a1, out, z1,z2,E, o)
    W1, W2 = GD(W1,W2,dW1,dW2)
fig, ax = plt.subplots()
plt.plot(Error)
plt.xlabel('number of iteration')
plt.ylabel('MSE')
plt.title('Mean squared error as a function of iteration')