from numpy import matrix, dot, transpose, multiply

#Variables known from the book
observation_model = [matrix('0.1 0; 0 0.8'), matrix('0.9 0; 0 0.2')]
transition_model = matrix('0.7 0.3; 0.3 0.7')
start_belief = matrix('0.5; 0.5')
backward_belief = matrix('1.0; 1.0')

def forward_backward(evidence):
    #Start by adding what we know
    fw_messages = [start_belief]
    bw_messages = [backward_belief]
    prior_fw = start_belief
    prior_bw = backward_belief
    #Loop to go through all the evidence
    for evidence_day in evidence:
        #Running forwardpart and appending to fw_messages
        prior_fw = forward(prior_fw, evidence_day)
        fw_messages.append(prior_fw)
        #Running backwardpart and appending to bw_messages
        prior_bw = backward(prior_bw, evidence_day)
        bw_messages.append(prior_bw)
    #Smoothing based on fw_messages and reversed bw_messages
    for i in range(len(fw_messages)):
        print normalize(smoothing(fw_messages[i],bw_messages[len(bw_messages)-i-1]))

#Backward part, dot product between transition model transposed, observation model based on evidence and prior belief
def backward(prior, evidence):
    evidence_based = dot(transition_model.getT(),observation_model[evidence])
    return normalize(dot(evidence_based, prior))

#Forward part, dot product between transition model, observation model based on evidence and prior belief
def forward(prior, evidence):
    probability = dot(transition_model,prior)
    return normalize(dot(observation_model[evidence],probability))

#Normalize matrixes
def normalize(evidence_based):
    return evidence_based/sum(evidence_based)

#Smooth matrixes by multiplying
def smoothing(fw_message, bw_message):
    return (multiply(fw_message,bw_message))

#Testing only forward part for task B
def forward_testing(evidence):
    prior = start_belief
    for evidence_day in evidence:
            print prior
            prior = forward(prior, evidence_day)
    return prior

#Evidence when umbrella appeard both days, 1=True
evidence1 = [1,1]
#Evidence for five days, 1=True, 0=False
evidence2 = [1,1,0,1,1]

print 'Part B1:'
print forward_testing(evidence1)

print 'Part B2'
print forward_testing(evidence2)

print 'Part C1'
forward_backward(evidence1)

print 'Part C2'
forward_backward(evidence2)


