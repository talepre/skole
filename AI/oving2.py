import numpy
import time

startTime=time.time()

#The dynamic model, defines the probability of whether or not there will be rain, given if there
#was rain or not the last day
T=numpy.matrix("0.7 0.3; 0.3 0.7")

#The observation model, split into two matrixes. Evidence used for input on whether or not there
#was an umbrella spotted.

umbrella=numpy.matrix("0.9 0; 0 0.2")
notUmbrella=numpy.matrix("0.1 0; 0 0.8")

#The initial state, there is an equal probability for rain or not.
startState=numpy.matrix('0.5;0.5')

#Structures for holding the messages being sent.
fMessages={}
bMessages={}
sMessages=[]

#The forward part of the algorithm
def FORWARD(E, state):	
	if state==0:
		fMessages[state]=startState
		return startState

	# Creating the message, using the dynamic model and previous days.
	message=E[state-1]*T*FORWARD(E, state-1)
	
	#normalize the message and save it in the list
	message=normalize(message)
	fMessages[state]=message	
	return message

#The backward part of the algorithm.
def BACKWARD(E, state):
	if state==len(E):
		bMessages[state]=numpy.matrix('1; 1')
		return numpy.matrix('1; 1')
	#Creating the message, using the dynamic model and previous days.
	message=E[state]*T*BACKWARD(E, state +1)
	bMessages[state]=message
	return message

def smoothing(E):
	FORWARD(E, len(E))
	BACKWARD(E, 0)
	#The part for computing the smoothed values for the hidden variables.
	for i in range(len(E)+1):
		num1=fMessages[i].flat[0]*bMessages[i].flat[0]
		num2=fMessages[i].flat[1]*bMessages[i].flat[1]
		message=numpy.matrix('%f;%f'%(num1,num2))
		sMessages.append(normalize(message))

#normalizing the elements in the matrix
def normalize(matrix):
	summed=sum(matrix.flat)
	for x, value in enumerate(matrix.flat):
		matrix.flat[x] = value/summed
	return matrix

def printMatrix(messageList):
	if type(messageList) is list:
		count=1
		for element in messageList:
			print("Iteration " + str(count) + " : ")
			print(element)
			count+=1
	else:
		for element in messageList:
			print("Iteration " + str(element + 1) + " : ")
			print(messageList[element])


print("\n")
print("Task B1, calculating probability of rain over two days, given umbrella being present both days: \n")
E1 = (umbrella, umbrella)
FORWARD(E1, 2)
printMatrix(fMessages)

print("\n")
print("Task B2, calculating probability of rain over five days, given umbrella, umbrella, not umbrella, umbrella, umbrella: ") 
fMessages={}
E2 = (umbrella, umbrella, notUmbrella, umbrella, umbrella)
FORWARD(E2, 5)
printMatrix(fMessages)

print("\n")
print("Task C1, same as task B, with smoothing added: ")
smoothing(E1)
printMatrix(sMessages)

print("\n")
print("Task C2: ")
fMessages={}
bMessages={}
sMessages=[]
smoothing(E2)
printMatrix(sMessages)

print("\n")
print("Execution: --- %s seconds ---" % str(time.time() - startTime))