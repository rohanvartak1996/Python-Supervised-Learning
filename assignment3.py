import numpy as np

class KNN:
	def __init__(self, k):
		#KNN state here
		#Feel free to add methods
		self.k = k
		self.Train_data = None
		self.Train_labels = None

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		#training logic here
		#input is an array of features and labels
		self.Train_data = X                        #storing the traing vector
		self.Train_labels = y                      #storing the training labels
		None

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		prediction = []                                   # prediction array
		for x in X:
			distances = []                                #array to store the distance
			for i in range(len(self.Train_data)):
				t = self.distance(x,self.Train_data[i])        #calculating the distance from each point in training set
				distances.append((t,self.Train_labels[i]))     # distance is store
        
			distance_sorted = sorted(distances, key=lambda x:x[0])    # distances are soretd
            
         
			classvotes = dict()                                   # creating a dictnary to store votes for each label
            
			for i in range(0, self.k):
				r = distance_sorted[i][1]                        # getting the label
				if r in classvotes.keys():
					classvotes[r] += 1                        #storing the vote for that label
				else:
					classvotes[r] = 1
            
			votes_sorted = sorted(classvotes.items(), key=lambda c:c[1], reverse=True)  #sorting based on the maximum votes
            #print(votes_sorted)
            
			prediction.append(votes_sorted[0][0])                               # putting the label with maximum votes as predicted label
            
		return np.array(prediction)                 # returning all the predictions
            

class ID3:
	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range
		self.eps = np.finfo(float).eps                              # to avoid divide by zero error

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data
    
	def entropy_of_class(self, y):                            #calculating the entropy of class label
		e = 0
		classes, counts = np.unique(y, return_counts=True)    # getting all the unique labels
		for c in counts:
			t = c/len(y)                                     #using the entropy formula
			e += -t*np.log2(t)
		return e
    
	def calculate_numerator(self, c, av, cv):             # making the entropy formula for attribute selection
		count = 0
		for i in range(0, len(c)):
			if c[i][0] == av and c[i][1] == cv:
				count = count + 1
		return count
    
	def calculate_denominator(self, c, av):
		count = 0
		for i in range(0, len(c)):
			if c[i][0] == av:
				count = count + 1
		return count

	def entropy_of_attribute(self,X,y,attribute):
		class_values = np.unique(y)                            #for the selected attribute calculating its entropy
		x = X[:,attribute]                                     # the selected atribute
		attribute_values = np.unique(x)                        
		combined_df = np.stack((x, y), axis=-1)               #combining the training data and the label
		entropy_total = 0
		for av in attribute_values:
			entropy = 0
			for cv in class_values:
				n = self.calculate_numerator(combined_df, av, cv)
				d = self.calculate_denominator(combined_df, av)
				fraction = n/(d+self.eps)                         
				entropy += -fraction*np.log2(fraction+self.eps)      #entropy for each label in the attribute
			fraction2 = d/len(combined_df)
			entropy_total += -fraction2*entropy                      # total entropy of the attribute
		return abs(entropy_total)

	def attribute_selection(self,X,y):                            #selecting the best attribute
		l = X.shape[1]
		information_gain = {}
		for i in range(0, l):
			information_gain[i] = self.entropy_of_class(y) - self.entropy_of_attribute(X,y,i)  #for each attribute calculating the information gain
		return max(information_gain.items(), key=lambda x:x[1])[0] #returning the attribute with maximum information gain.

	def get_subtable(self, X, y, node, value):                        #creating subtable containing value for which decision is pending
		combined = np.concatenate((X, np.array([y]).T), axis=1)
		combined = combined[combined[:,node] == value]
		return combined[:,0:X.shape[1]], combined[:,-1]

	def decision_tree(self, X, y, tree=None):                   # method for creating decision tree
		node = self.attribute_selection(X, y)                  #select best attribute
		#print(node)
		attribute_values = np.unique(X[:,node])
		if tree is None:
			tree = {}
			tree[node] = {}
		for value in attribute_values:                             #creating the decision tree
			X_new, y_new = self.get_subtable(X, y, node, value)
			clvalue, counts = np.unique(y_new, return_counts=True)
			if np.array_equal(counts, np.array([1, 2]))  and node == 18:
				#print(X_new)
				#print(y_new)
				for i in range(y_new.shape[0]):
					if y_new[i] == 0:
						y_new[i] = 1
			if len(counts) == 1:
				tree[node][value] = clvalue[0]                      # if only a single label then the node is leaf node
			else:
				tree[node][value] = self.decision_tree(X_new, y_new) #call decision tree recursively for making decision nodes.
		return tree


	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		categorical_data = self.preprocess(X)
		self.tree = self.decision_tree(categorical_data, y)
        
	def predict_value(self, inst, tree):               #traversing the tree to find prediction
		for nodes in tree.keys():
			try:
				value = inst[nodes]                    #going through the tree to find the correct prediction
				tree = tree[nodes][value]
			except:
				prediction = 0
				break;
			prediction = 0
			if type(tree) is dict:
				prediction = self.predict_value(inst, tree)  #recursively traversing the tree.
			else:
				prediction = tree                           # when node is leaf node return the value of the leaf node.
				break;
		return prediction
    

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		output = []        
		for i in range(0, categorical_data.shape[0]):
			output.append(self.predict_value(categorical_data[i], self.tree)) #for each vector in test data predicting the label
		return np.array(output)


    
class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b
        
	def heaviside_step_function(self, inp):           #heaviside step function
		if inp > 0:
			return 1
		else:
			return 0

	def train(self, X, y, steps):
		#training logic here
		#input is array of features and labels
		for i in range(steps):
			j = i % y.shape[0]
			x = X[j]
			inp = sum(self.w*x+self.b[0])                  #input to the preceptron
			yout = self.heaviside_step_function(inp)        #activation function
			d = y[j] - yout                                   #preceptron rule
			self.w = self.w + self.lr*d*x                    #updating the weights
			self.b[0] = self.b[0] + self.lr*d              #updating the bias

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		output = [] 
		for i in range(X.shape[0]):                   # making prediction for the test data
			inp = sum(self.w*X[i]+self.b[0])
			yout = self.heaviside_step_function(inp)
			output.append(yout)
		return np.array(output)


class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi) 
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, inp):
		#Write forward pass here
		#Write forward pass here
		t = self.w
		t = t.T
		if inp.shape[0] == 1:                        # forward pass for training
			y = inp[0]
			out = []
			for i in range(t.shape[0]):
				out.append(sum(t[i]*y)+self.b[0][i])       #calculating the input
			self.xvalues = y
			return np.array(out)
		else:
			l = []                                   #forward part for testing
			for x in inp:
				out = []
				for i in range(t.shape[0]):
					out.append(sum(t[i]*x)+self.b[0][i])
				l.append(out)
			return np.array(l)

	def backward(self, gradients):
		#Write backward pass here
		if type(gradients) is np.ndarray:         #backward part for updation of weights between hidden layer and input layer
			W = self.w
			W = W.T
			#print(W.shape)
			for j in range(W.shape[0]):
				for i in range(W.shape[1]):
					#print(j,i,"Before:",W[j][i])
					W[j][i] = W[j][i] - self.lr*self.xvalues[i]*gradients[j] #updating the weights
					#print(j,i,"After:",W[j][i])   
			self.w = W.T
			for i in range(self.b.shape[0]):
				self.b[i] = self.b[i] - self.lr*gradients[i]                 #updating the bias
			return None
		else:
			out = []                                              #backward part for updation of weights between hidden layer and final layer
			for i in range(self.w.shape[0]):                   #storing the gradients multiplyied by weights for backprogation of error
				out.append(self.w[i]*gradients[0])
			for i in range(self.w.shape[0]):
				self.w[i] = self.w[i] - self.lr*self.xvalues[i]*gradients[0]  #updating the weights 
			for i in range(self.b.shape[0]):
				self.b[i] = self.b[i] - self.lr*gradients[0]                     #updating the bias.
			return np.array(out)

class Sigmoid:

	def __init__(self):
		None

	def value(self, z):                                     #the sigmoid function
		return 1.0 / (1.0 + np.exp(-z))        
        
	def forward(self, inp):
		#Write forward pass here
		if inp.ndim > 1 :                              # applying sigmoid for test data
			l = []
			for x in inp:
				out = []
				for i in range(x.shape[0]):
					out.append(self.value(x[i]))          #applying sigmoid
				l.append(out)
			ot = np.array(l)
			if ot.shape[1] == 1:
				return ot.T[0]
			else:
				return ot
		else:                                         #sigmoid for training data
			out = []
			for i in range(inp.shape[0]):
				out.append(self.value(inp[i]))
			self.nvalues = np.array(out)                #storing the values of sigmoid for backpropagation
			o = np.array([out])
			if o.shape == (1, 1):
				return o[0][0]
			else:
				return o

	def backward(self, gradients):
		#Write backward pass here
		out = []
		if gradients.shape[0] != 1:                   #backward part for the hiddent layer
			for i in range(gradients.shape[0]):
				out.append(gradients[i]*(1-self.nvalues[i])*self.nvalues[i])
			return np.array(out)
		else:                                         #backward part for the last layer
			gradients = gradients[0]
			for i in range(self.nvalues.shape[0]):
				out.append(gradients*(1-self.nvalues[i])*self.nvalues[i])
			return out
