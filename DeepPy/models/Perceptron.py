import numpy as np

class Perceptron:

	def train(self,x,y,learningRate,maxIter):

		lenWeights = len(x[1,:]);
		weights = np.random.rand(lenWeights);
		bias = np.random.random();
		t = 1;
		converged = False;

		# Perceptron Algorithm

		while not converged and t < maxIter:
			targets = [];
			for i in range(len(x)):

				# Calculate output of the network
				# The decision function is given by the line w'x + b = 0;
				output = np.dot(x[i,:],weights) + bias;

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
				weights = weights + (x[i,:] * (learningRate * error));
				bias = bias + (learningRate * error);
				targets.append(target);

				t = t + 1;

		if ( list(y) == list(targets) ) == True:
			# As soon as a solution is found break out of the loop
			converged = True;


		model = {"weights":weights, "bias":bias};
		return model

	def test(self,x, model):

		predictions = [];
		margins = [];
		weights = model['weights'];
		bias = model['bias'];

		for i in range(len(x)):
		
			# Calculate w'x + b
			output = np.dot(x[i,:],weights) + bias;
		
			# Get decision from hardlim function
			if (output > 0):
				target = 1;
			else:
				target = 0;

			predictions.append(target);

		return predictions
