import operator 
import math
import copy

class TDIDT:
    def __init__(self,t):
        self.name = "NAN"
        self.target = t
        self.children = {}

    def put_dataset(self,ds,wrap):
        """
        Given a list of dictionaries, create a node and branch with Shannon Entropy 
        """
        selected = self.get_Att_Selection(ds,wrap)
        self.name = selected
        if self.name != self.target:
            self.name = selected
            self.children = self.parition_on_Att(ds,selected,wrap)
        else:
            for x in ds:
                if x[self.target] not in self.children:
                    self.children[x[self.target]] = 1
                else:
                    self.children[x[self.target]] += 1
        return self

    def parition_on_Att(self,ds,a,wrap):
        """
        Given a dataset (list of dictionaries), a selected attribute to split the dataset on, and a 
        list of relevant attributes, create children for the attributes different values
        """
        att_lists = {}
        result = {}
        for x in ds:
            if x[a] not in att_lists:
                att_lists[x[a]] = [x]
            else:
                att_lists[x[a]].append(x)
        for y in att_lists.keys():
            tmp = TDIDT(self.target)
            result[y] = tmp.put_dataset(att_lists[y],wrap)
        return result
        
    def get_Att_Selection(self,ds,wrap):
        """
        Given a dataset, calculate the attribute that will give the greatest Information 
        Gain, and return it so that the dataset may be partitioned
        """
        keys = ds[0].keys()
        keys.remove(self.target)
        if wrap:
            keys = wrap[:]
        result = {}
        for a in keys:
            result[a] = self.calculate_En(ds,a)
        if len(set(result.values()))==1:
            return self.target
        else:
            return min(result.iteritems(),key=operator.itemgetter(1))[0]

    def calculate_En(self,ds,a):
        """
        Calculate an attributes Enew value over the dataset
        """
        val = {}
        result = 0
        for x in ds:
            if x[a] not in val:
                val[x[a]] = 1
            else:
                val[x[a]] += 1
        total = len(ds)
        for v in val.keys():
            result += (float(val[v])/float(total)) * self.calculate_E(ds,a,v)
        return result

    def calculate_E(self,ds,a,v):
        """
        Calculate the Entropy for a specific value of a dataset
        """
        cls = {}
        result = 0
        for x in ds:
            if (x[self.target] not in cls and x[a] == v):
                cls[x[self.target]] = 1
            elif x[a] == v:
                cls[x[self.target]] += 1
        total = sum(cls.itervalues())
        for v in cls.keys():
            result -= (float(cls[v])/float(total)) * math.log((float(cls[v])/float(total)),2)
        return result

    def condense(self, node):
        """
        Reduce the tree, so that only the nessessary nodes are left. Does not
        affect unique paths. NO loss to information in the tree. 
        """
        if node.name == self.target:
            return node
        all_leaves = True
        for key in node.children.keys():
            node.children[key] = self.condense(node.children[key])
            if node.children[key].name != self.target:
                all_leaves = False
        if all_leaves:
            leaves_all_same = True
            classifier = None
            for child in node.children.values():
                if classifier == None:
                    classifier = self.get_most_pop_classifer(child.children)
                elif classifier != self.get_most_pop_classifer(child.children):
                    leaves_all_same = False
                    break
            if leaves_all_same:
                outcomes = None
                for child in node.children.values():
                    if outcomes == None:
                            outcomes = child.children
                    else:
                        for outcome in child.children.keys():
                            try:
                                outcomes[outcome] += child.children[outcome]
                            except KeyError:
                                outcomes[outcome] = child.children[outcome]
                out = TDIDT(self.target)
                out.name = self.target
                out.children = outcomes
                return out
        return node

    def get_best_condensable_tree(self, trees):
        """
        Given a list of trees, returns the tree that condenses the best
        """
        condensed_tree_sizes = []
        for tree in trees:
            condensed_tree_sizes.append(len(tree.condense(tree).view_tree()))
        i = condensed_tree_sizes.index(min(condensed_tree_sizes))
        return trees[i]


    def get_most_pop_classifer(self, c):
        classifier = None
        for key in c.keys():
            if classifier == None or c[key] > c[classifier]:
                classifier = key
        return classifier

    def classify(self,inst):
        """
        given an instance, traverse the tree to find the most likely class label
        """
        if self.name == self.target:
            #handle empty children!
            classification = max(self.children.iteritems(), key=operator.itemgetter(1))[0]
            return classification
        else:
            if inst[self.name] in self.children:
                return self.children[inst[self.name]].classify(inst)
            else:
                #handle the case that it is not in children!!
                return None
        
    def prt(self, depth):
        """
        Print the Tree! Now with nice format - Thanks nate!!
        """
        out = ''
        indent = ""
        for i in range(0, depth):
            indent += '  '
        if self.target == self.name:
            out += indent + str(self.children) + '\n'
        else: 
            for child in self.children.keys():
                out += indent + str(self.name) + ':' + str(child) + '\n'
                out += self.children[child].prt(depth+1)
        return out

    def print_rules(self):
        """
        Rule printer helper. Super handy 
        """
        return self.string_rules(0,'')

    def string_rules(self,depth,out):
        """
        Prints all of the paths of the tree as "Rules"
        """
        tmp_out = out
        if self.target == self.name:
            print out + " THEN class == " + str(max(self.children.iteritems(), key=operator.itemgetter(1))[0])
        else: 
            for child in self.children:
                tmp_out += "if " + str(self.name) + ' == ' + str(child)
                if self.target not in self.children[child].name:
                    tmp_out += " AND "
                    self.children[child].string_rules(depth+1,tmp_out)
                else:
                    self.children[child].string_rules(depth+1,tmp_out)
                tmp_out = out

    def generate_dot_file(self,filename):
        """
        GExtra Credit attempt. Does not actually work right now
        """
        with open(fileout, "wb") as out:
            out.write("graph g{ \n")
            out.write(self.dot_helper(0))
            out.write("}\n")

    def dot_helper(self,count):
        count = 0
    
    def __str__(self): 
        return self.prt(0)
