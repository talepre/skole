def forward_backward(evidence, prior):
	forward_vector = [0,0,0]
	backward_message = [0,0,0]
	smooth_vector = [0,0,0]
	
	forward_vector[0] = prior;
	for t in range(1,len(smooth_vector)):
		forward_vector[1] = forward(forward_vector(t-1),evidence[t])



	return smooth_vector

def forward():
	



'''
function FORWARD-BACKWARD(ev, prior ) returns a vector of probability distributions
inputs: ev, a vector of evidence values for steps 1,...,t
prior , the prior distribution on the initial state, P(X0)
local variables: fv, a vector of forward messages for steps 0,...,t
b, a representation of the backward message, initially all 1s
sv, a vector of smoothed estimates for steps 1,...,t
fv[0] ← prior
for i = 1 to t do
fv[i] ← FORWARD(fv[i − 1], ev[i])
for i = t downto 1 do
sv[i] ← NORMALIZE(fv[i] × b)
b← BACKWARD(b, ev[i])
return sv
'''

'''
function FIXED-LAG-SMOOTHING(et, hmm, d) returns a distribution over Xt−d
inputs: et, the current evidence for time step t
hmm, a hidden Markov model with S × S transition matrix T
d, the length of the lag for smoothing
persistent: t, the current time, initially 1
f, the forward message P(Xt|e1:t), initially hmm.PRIOR
B, the d-step backward transformation matrix, initially the identity matrix
et−d:t, double-ended list of evidence from t − d to t, initially empty
local variables: Ot−d, Ot, diagonal matrices containing the sensor model information
add et to the end of et−d:t
Ot ← diagonal matrix containing P(et|Xt)
if t>d then
f← FORWARD(f, et)
remove et−d−1 from the beginning of et−d:t
Ot−d ← diagonal matrix containing P(et−d|Xt−d)
B← O−1
t−dT−1BTOt
else B← BTOt
t ← t + 1
if t>d then return NORMALIZE(f × B1) else return null
'''