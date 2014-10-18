
        
class Tree:
    def __init__(self, node_label, node_value):
        self.node_label = node_label
        self.node_value = node_value
        self.subtrees = []
        
    def append(self, subtree):
        self.subtrees.append(subtree)

    def __str__(self): 
        result = '(Node '
        result += '<' + str(self.node_label) + ':' + str(self.node_value) + '>'
        result += ' ('
        for s in self.subtrees[:-1]:
            result += str(s) + ' '
        if len(self.subtrees) > 0:
            result += str(self.subtrees[-1])
        return result + '))'


def main():
    t = Tree('attribute', 0)
    t.append(Tree('value', 1))
    t.append(Tree('value', 2))
    t.append(Tree('value', 3))
    t.subtrees[0].append(Tree('attribute', 1))
    t.subtrees[0].subtrees[0].append(Tree('value', '1'))
    t.subtrees[0].subtrees[0].subtrees[0].append(Tree('yes', [1, 1]))
    print t


if __name__ == '__main__':
    main()

