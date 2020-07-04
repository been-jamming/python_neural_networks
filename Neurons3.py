import math
import random

def Sigmoid(input):
	try:
		return 1/(1+math.exp(-input))
	except OverflowError:
		if input > 0:
			return 1
		else:
			return -1

def SigmoidDerivative(input):
	return ActivateFunc(input)*(1-ActivateFunc(input))

def Cost(output, expected):
	return 0.5*(output - expected)**2

def CostDerivative(output, expected):
	return (output - expected)*ActivateDerivative(output)

def CreateNetwork(structure, ActFunc = False, ActDeriv = False):
	layerlist = []
	for layer in structure:
		neuronlist = []
		for neuron in layer:
			neuronlist.append(Neuron([0.5]*neuron,0.5, ActFunc, ActDeriv))
		layerlist.append(Layer(neuronlist))
	return Network(layerlist)

class Neuron():
	def __init__(self, weights, bias, ActFunc = False, ActDeriv = False):
		self.weights = weights
		self.lastweights = weights
		self.bias = bias
		self.lastsum = 0
		self.lastchange = 0
		self.ActFunc = ActFunc
		self.ActDeriv = ActDeriv
	def Push(self, inputs):
		self.lastinputs = inputs
		sum = 0
		for i in range(len(inputs)):
			sum += inputs[i]*self.weights[i]
		sum += self.bias
		self.lastsum = sum
		if not self.ActFunc:
			sum = 1/(1+math.exp(-sum))
		else:
			sum = self.ActFunc(sum)
		self.lastout = sum
		return sum
	def Teach(self, expected, rate, m=0,  sum = False):
		self.lastweights = self.weights
		if sum == False:
			sum = self.lastout - expected
		if not self.ActDeriv:
			deriv = self.lastout*(1-self.lastout)*sum
		else:
			deriv = self.ActDeriv(self.lastsum)*sum
		change = deriv*rate + self.lastchange*m
		for weight in range(len(self.weights)):
			self.weights[weight] -= self.lastinputs[weight]*change
		self.bias -= change
		self.lastchange = change
		self.deriv = deriv
	def RandomWeights(self, lower = -1, upper = 1):
		for w in range(len(self.weights)):
			self.weights[w] = random.uniform(lower, upper)
	def RandomBias(self, upper = 1, lower = -1):
		self.bias = random.uniform(lower, upper)

class Layer():
	def __init__(self, neurons):
		self.neurons = neurons
	def Push(self, inputs):
		output = []
		for input in range(len(inputs)):
			output.append(self.neurons[input].Push(inputs[input]))
		self.lastoutput = output
		return output
	def Teach(self, expected, rate):
		for expect in range(len(expected)):
			self.neurons[expect].Teach(expected[expect], rate)
	def RandomWeights(self, lower = -1, upper = 1):
		for neuron in self.neurons:
			neuron.RandomWeights(lower, upper)
	def RandomBias(self, lower = -1, upper = 1):
		neuron.RandomBias(lower, upper)

class Network():
		def __init__(self, layers):
			self.layers = layers
		def Push(self, inputs):
			self.lastinput = inputs
			for layer in range(len(self.layers)):
				output = self.layers[layer].Push(inputs)
				try:
					inputs = len(self.layers[layer+1].neurons)*[output]
				except:
					self.lastoutput = output
					return output
		def Teach(self, expected, rate, m = 0):
			for i in list(reversed(range(len(self.layers)))):
				for j in range(len(self.layers[i].neurons)):
					if i == len(self.layers)-1:
						self.layers[i].neurons[j].Teach(expected[j], rate, m)
					else:
						sum = 0
						for neuron in self.layers[i+1].neurons:
							sum += neuron.deriv*neuron.weights[j]
						self.layers[i].neurons[j].Teach(0, rate, m, sum)
		def RandomWeights(self, lower = -1, upper = 1):
			for layer in self.layers:
				layer.RandomWeights(lower, upper)
		def RandomBias(self, lower = -1, upper = -1):
			layer.RandomBias(lower, upper)