from numpy import matrix, dot, around, transpose, multiply

observation_model = [matrix('0.1 0; 0 0.8'), matrix('0.9 0; 0 0.2')]
transition_model = matrix('0.7 0.3; 0.3 0.7')
start_belief = matrix('0.5; 0.5')
backward_belief = matrix('1.0; 1.0')

def forward_backward():
    return None

def backward():
    return None

def forward(transition_model, prior, evidence):
    return normalize((transition_model*prior), evidence)

def normalize(probability, evidence):
    evidence_based = (observation_model[evidence]*probability)
    return evidence_based/sum(evidence_based)

def forward_testing(transition_model, evidence):
    prior = start_belief
    for evidence_day in evidence:
            print around(prior, decimals=3)
            prior = forward(transition_model, prior, evidence_day)
    return prior


evidence1 = [1,1]
print 'Part B1:'
print around(forward_testing(transition_model, evidence1), decimals=3)
evidence2 = [1,1,0,1,1]
print 'Part B2'
print around(forward_testing(transition_model, evidence2), decimals=3)

print 'test'
first = (multiply(matrix('0.818; 0.182'),matrix('0.69; 0.41')))
second = first/sum(first)
print around(second, decimals=3)