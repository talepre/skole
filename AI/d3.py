from random import randint
from math import log

class Node():
	def __init__(self, data):
		self.data = data
		self.children = {}
	
	def addBranch(self, key, value):
		self.children[key] = value

        #Function to print this node and all children recursivly
	#Can be plotted here: http://ironcreek.net/phpsyntaxtree
	def getTreeStructure(self):
		if len(self.children) == 0:
			return "[" + str(self.data) + "]"
		else:
			temp = "[" + str(self.data) + " "
		for k, v in self.children.items():
			temp += self.children[k].getTreeStructure()
		return temp + "]"
	


#Read training/test file
def readfile(filename):
	f = open("data/" + filename + ".txt")
	return [w.split("	") for w in f.read().split("\n")]


#Get boolean entropy
def B(q):
        if q == 0: return 0
        return -(q*log(q, 2) + ((1.0-q)*log(1.0-q, 2)))


#Find most common classification category
def plurality(data):
	counter = {}
	
	for e in data:
		if e[-1] in counter:
			counter[e[-1]] += 1
		else: 
			counter[e[-1]] = 1
	return max(counter, key=counter.get)


#Returns true if all classifications of the data are the same
def sameClassification(data):
	for line in data:
		if line[-1] != data[0][-1]: return False
	return True


#Returns attribute to split on
def randomImportance(data, attributes):
        return attributes[randint(0, len(attributes)-1)]


#Returns attribute to split on, assumes boolean classification
def gainImportance(data, attributes):
        entropy = {}
        for a in attributes:
                counter = 0
                for line in data:
                        if line[a] == data[0][a]:
                                counter += 1

                entropy[a] = B(counter/len(data))
        return min(entropy, key=entropy.get)


#Classify a single entry given the root of the classification tree
def classify(root, line):
        current = root
        while current.children:
                current = current.children[int(line[current.data])]
        return current.data


#Returns root Node of decision tree, assumes boolean classification
def decisionTreeLearning(examples, attributes, parentExamples, rndImp):
	if not examples:
		return Node(plurality(parentExamples))
	elif sameClassification(examples):
		return Node(examples[0][-1])
	elif not attributes:
		return Node(plurality(examples))
	else:
		if rndImp:
			A = randomImportance(examples, attributes)
		else:
			A = gainImportance(examples, attributes)
                        
		tree = Node(A)
		attributes.remove(A)

		for vk in xrange(1, 3):
			exs = [e for e in examples if int(e[A]) == vk]
			subtree = decisionTreeLearning(exs, list(attributes), examples, rndImp)
			tree.addBranch(vk, subtree)
	return tree


#Run classifications on data and calculate how many correct results
def runTests(tree, data):
        correct = 0
        for line in data:
                if line[-1] == classify(tree, line):
                        correct += 1
        print "Correct", correct, "out of", len(data), "Rate:", float(correct)/len(data)


def main():
        trainingdata = readfile("training")
        testdata = readfile("test")

        for i in xrange(2):
                root = decisionTreeLearning(trainingdata, range(7), [], i==0)
                print root.getTreeStructure()
                runTests(root, testdata)
                runTests(root, trainingdata)
                print

main()