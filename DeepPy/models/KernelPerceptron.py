import numpy as np
from DeepPy.kernels.rbf import rbf

class KernelPerceptron:

	def train(self,x,y,sigma,learningRate,maxIter):

		K = rbf(x, sigma);
		lenWeights = len(K[1,:]);
		weights = np.random.rand(lenWeights);
		bias = np.random.random();
		t = 1;
		converged = False;

		# Kernel Perceptron Algorithm

		while not converged and t < maxIter:
			targets = [];
			for i in range(len(x)):

				# Calculate output of the network
				# Kernel method here allows for non-linear classifier
				output = np.dot(K[i,:],weights) + bias;

				# Perceptron threshold decision: 
				# If w'x[i,:] + b > 0 then the output of x[i,:] is 1
				# If w'x[i,:] + b < 0 then the output of x[i,:] is 0
				if (output > 0):
					target = 1;
				else:
					target = 0;

				# Calculate error and update weights
				# Shifts the decision boundary
				error = y[i] - target;
				weights = weights + ((K[i,:] * learningRate * error));
				bias = bias + (learningRate * error);
				targets.append(target);

				t = t + 1;

		if ( list(y) == list(targets) ) == True:
			# As soon as a solution is found break out of the loop
			converged = True;

		model = {"weights":weights,"bias":bias,"sigma":sigma}
		return model

	def test(self,x, model):

		predictions = [];
		margins = [];
		sigma = model["sigma"];
		weights = model["weights"];
		bias = model["bias"];
		K = rbf(x, sigma);

		for i in range(len(x)):
		
			# Calculate w'x + b
			output = np.dot(K[i,:],weights) + bias;
			margins.append(output);
		
			# Get decision from hardlim function
			if (output > 0):
				target = 1;
			else:
				target = 0;

			predictions.append(target);

		return predictions
