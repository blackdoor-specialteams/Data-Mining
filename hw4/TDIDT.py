
class TDIDT:
    def __init__(self,n,t):
        self.name = n
        self.target = t
        self.children = {}
        
    def append(self,val,att):
        self.children.update({val:TDIDT(att,self.target)})
    
    def put_row(self,data,attributes):
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
            node.children[key] = condense(node.children[key])
            if node.children[key].name != self.target:
                all_leaves = False
        if all_leaves:
            leaves_all_same = True
            classifier = None
            for child in node.children:
                if classifier == None:
                    classifier = get_most_pop_classifer(child)
                elif classifier != get_most_pop_classifer(child):
                    leaves_all_same = False
                    break
            if leaves_all_same:
                outcomes = None
                for child in node.children:
                    if outcomes == None:
                            outcomes = child.children
                    else:
                        for outcome in child.children.keys():
                            outcomes[outcome] += child.children[outcome]
                out = TDIDT.TDIDT(self.target, self.target)
                out.children = outcomes
                return out
        return node

    def get_most_pop_classifer(c):
        classifier = None
        for key in c.keys():
            if classifier == None or c[key] > c[classifier]:
                classifier = key
        return classifier

    def classify(self,inst):
        return None

    def view_tree(self):
        return self.prt(0)

    def prt(self, depth):
        out = ''
        indent = ""
        for i in range(0, depth):
            indent += '  '
        if self.target == self.name:
            out += indent + str(self.children) + '\n'
        else: 
            for child in self.children.keys():
                out += indent + self.name + ':' + child + '\n'
                out += self.children[child].prt(depth+1)
        return out

    def __str__(self): 
        result = ' <Att: '
        result += '' + str(self.name) + '>'
        result += ' ('
        for k in self.children.keys():
            result += " " + str(k)+ "." +  str(self.children[k]) + "\n"
        return result + ')'
