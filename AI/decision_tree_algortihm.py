from random import randint, choice
from math import log

class Node(object):
    def __init__(self, atr):
        self.atr = atr
        self.childNodes = {}

def print_tree(node):
    nodes = list()
    nodes.append(node)
    while len(nodes) > 0:
        atrs = []
        new_nodes = []
        for n in nodes:

            if len(list(n.childNodes.keys())) != 0:
                atrs.append(str(n.atr))
                for key in sorted(list(n.childNodes.keys())):
                    new_nodes.append(n.childNodes[key])
            else:
                atrs.append("class: " + str(n.atr))
        print(atrs)
        nodes = new_nodes

def get_file_list(filename):
    f = open(filename, 'r').read().split('\n')
    for i in range(len(f)):
        f[i] = f[i].split('\t')
    return f


def all_instances_has_same_categorization(training_file):
    for line in training_file:
        if line[-1] != training_file[0][-1]:
            return False
    return True

def get_random_importance(attributes):
    return choice(attributes)

#calculate expected information gain based on the entropy. Returns min.
def get_expected_information_gain_importance(attributes, training_file):
    #using a number guaranteed to be higher, as we look for the lowest anyway
    counting_values_list = [2 for v in init_attributes]
    for atr in attributes:
        c = 0
        for line in training_file:
            if line[atr] == training_file[0][atr]:
                c += 1
        q = c/len(training_file)
        if q == 0:
            counting_values_list[atr] = 0
        else:
            #Calculating the value of B(q) (entropy of boolean variable)
            counting_values_list[atr] = -(q*log(q, 2) + ((1.0-q)*log(1.0-q, 2)))
    return counting_values_list.index(min(counting_values_list))

#returns index of the best attribute for classification
def get_plurality_value(training_file):
    counting_values_list = [0 for v in values]
    for line in training_file:
        for i in range(len(values)):
            if int(line[-1]) == values[i]:
                counting_values_list[i] += 1
    return counting_values_list.index(max(counting_values_list))



#Recursively creates the decision tree.
def create_decision_tree(training_file, attributes, parent_training_file, random_importance):
    if len(training_file) == 0:
        return Node(get_plurality_value(parent_training_file))
    elif all_instances_has_same_categorization(training_file):
        #All nodes point to category
        return Node(training_file[0][-1])
    elif len(attributes) == 0:
        return Node(get_plurality_value(training_file))
    else:
        if random_importance:
            A = get_random_importance(attributes)
        else:
            A = get_expected_information_gain_importance(attributes, training_file)
        tree = Node(A)
        attributes.remove(A)
        for v in values:
            exs = []
            for e in training_file:
                if int(e[A]) == v:
                    exs.append(e)
            subtree = create_decision_tree(exs, list(attributes), training_file, random_importance)
            tree.childNodes[v] = subtree
        return tree

def classify_tests(root_node, test_file):
    c = 0
    for i in range(len(test_file)):
        node = root_node
        while len(node.childNodes) != 0:
            node = node.childNodes[int(test_file[i][node.atr])]
        #print("line " + str(i+1) +" classified as: " + node.atr)
        if test_file[i][-1] == node.atr:
            #print("Right")
            c += 1
        #else:
        #    print("Wrong")
    print("Score: " + str(float(c/len(test_file))))


training_file = get_file_list("training.txt")
test_file = get_file_list("test.txt")
init_attributes = [0, 1, 2, 3, 4, 5, 6]
values = [1, 2]

#With random importance
root_node = create_decision_tree(training_file, list(init_attributes), [], True)
print_tree(root_node)
classify_tests(root_node, test_file)

#with information gain importance
root_node = create_decision_tree(training_file, list(init_attributes), [], False)
print_tree(root_node)
classify_tests(root_node, test_file)

