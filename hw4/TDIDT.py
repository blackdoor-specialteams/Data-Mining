import operator 
class TDIDT:
    def __init__(self,t):
        self.name = "NAN"
        self.target = t
        self.children = {}
        
    def append(self,val,att):
        self.children.update({val:TDIDT(att,self.target)})
    
    def get_Att_En(ds):
        keys = ds[0].keys()
        result = {}
        for a in keys:
            result[a] = calculate_En(ds,a)
        return result

    def calculate_En(ds,a):
        val = {}
        result = 0
        for x in ds:
            if x[a] not in val:
                val[a] = 1
            else:
                val[a] += 1
        total = sum(val.values())
        for v in val.keys():
            result += (float(val[v])/float(total)) * calculate_E(ds,a,v)
        return result

    def calculate_E(ds,a,v):
        cls = {}
        result = 0
        for x in ds:
            if (x[self.target] not in cls) and (x[a] == v):
                cls[x[self.target]] = 1
            elif x[a] == v:
                cls[x[self.target]] += 1
        total = sum(cls.values())
        for v in cls.keys():
            result -= (float(cls[v])/float(total)) * math.log((float(cls[v])/float(total)),2)
        return result

    def put_dataset(ds):
        return None

    def parition_on_Att(ds,a):
        result = {}
        return None

    def partition_on_Att_Val(ds,a,v):
        return None

    def put_row(self,data,attributes):
        """
        last element in attributes MUST be target
        """
        #if no more attributes to insert, return
        if not attributes:
            print "Attribute list is NULL"
            return None
        elif len(attributes) > 1:
            #insert value node if there isnt one already for this attribute
            if data[attributes[0]] not in self.children.keys():
                self.append(data[attributes[0]],attributes[1])
            #recurse the node that has this attributes value
            self.children[data[attributes[0]]].put_row(data,attributes[1:])
        else:
            #length of list is 1, meaning this is the class label
            if data[attributes[0]] not in self.children.keys():
                stats = {data[attributes[0]]:1}
                self.children.update(stats)
            else:
                self.children[data[attributes[0]]] += 1

    def condense(self, node):
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
                out = TDIDT(self.target, self.target)
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
        return self.string_rules(0,'')

    def string_rules(self,depth,out):
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
        with open(fileout, "wb") as out:
            out.write("graph g{ \n")
            out.write(self.dot_helper(0))
            out.write("}\n")

    def dot_helper(self,count):
        count = 0
        

    def __str__(self): 
        return self.prt(0)
