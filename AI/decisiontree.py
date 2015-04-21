function DECISION-TREE-LEARNING(examples, attributes, parent examples) returns
a tree
if examples is empty then return PLURALITY-VALUE(parent examples)
else if all examples have the same classification then return the classification
else if attributes is empty then return PLURALITY-VALUE(examples)
else
A ←argmaxa ∈ attributes IMPORTANCE(a, examples)
tree ← a new decision tree with root test A
for each value vk of A do
exs ← {e : e ∈examples and e.A = vk}
subtree ← DECISION-TREE-LEARNING(exs, attributes − A, examples)
add a branch to tree with label (A = vk) and subtree subtree
return tree