
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
    
    def condense(self):
        return None

    def classify(self,inst):
        return None

    def __str__(self): 
        result = '(Att: '
        result += '<' + str(self.name) + '>'
        result += ' ('
        for k in self.children.keys():
            result += str(self.children[k])
        return result + '))'
