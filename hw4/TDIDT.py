
class TDIDT:
    def __init__(self, node_label, node_value):
        self.node_label = node_label
        self.node_value = node_value
        self.subtrees = []
        
    def append(self, subtree):
        self.subtrees.append(subtree)
    
    def put_row(self,data,attributes):
        #if no more attributes to insert, return
        if not attributes:
            print "Attribute list is NULL"
            return None
        elif len(attributes) > 1:
            #insert value node if there isnt one already for this attribute
            if data[attributes[0]] not in self.get_subtree_values():
                nnode = TDIDT('V',data[attributes[0]])
                self.append(nnode)
            #find the node that has this attributes value
            for s in self.subtrees[:-1]:
                if data[attributes[0]] == s.node_value:
                        if not s.subtrees:
                            s.append(TDIDT("ATT",attributes[1]))
                        s.subtrees[0].put_row(data,attributes[1:])
        else:
            #length of list is 1, meaning this is the class label
            if data[attributes[0]] not in self.node_value:
                stats = {"count":1,"total":1,"p":1}
                nnode = TDIDT(data[attributes[0]],stats)
                self.append(nnode)
            else:
                for s in self.subtrees[:-1]:
                    if s.node_label == data[attributes[0]]:
                        s.node_value["count"] += 1
            self.update_class_leafs()
    
    def condense(self):
        return None

    def classify(self,inst):
        return None

    def update_class_leafs(self):
        total = 0
        for n in self.subtrees[:-1]:
            total += n.node_value["count"]
        for n in self.subtrees[:-1]:
            n.node_value["total"] = total
            n.node_value["p"] = float(n.node_value["count"]) /float(total)

    def get_subtree_values(self):
        result = []
        for x in self.subtrees[:-1]:
            result.append(x.node_value)
        return result

    def get_subtree_labels(self):
        result = []
        for x in self.subtrees[:-1]:
            result.append(x.node_label)
        return result

    def __str__(self): 
        result = '(Node '
        result += '<' + str(self.node_label) + ':' + str(self.node_value) + '>'
        result += ' ('
        for s in self.subtrees[:-1]:
            result += str(s) + ' '
        if len(self.subtrees) > 0:
            result += str(self.subtrees[-1])
        return result + '))'
