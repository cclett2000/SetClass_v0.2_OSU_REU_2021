# this creates a crude data tree; from: https://medium.com/swlh/making-data-trees-in-python-3a3ceb050cfd

class Tree():
    def __init__(self, root):
        self.root = root
        self.children = []
        self.Nodes = []

    def addNode(self, obj):
        self.children.append(obj)

    def getAllNodes(self):
        self.Nodes.append(self.root)
        for child in self.children:
            self.Nodes.append(child.data)
        for child in self.children:
            if child.getChildNodes(self.Nodes) != None:
                child.getChildNodes(self.Nodes)
        print(*self.Nodes, sep='\n')
        print('Tree Size:' + str(len(self.Nodes)))


class Node():
    def __init__(self, data):
        self.data = data
        self.children = []

    def addNode(self, obj):
        self.children.append(obj)

    def getChildNodes(self, Tree):
        for child in self.children:
            if child.children:
                child.getChildNodes(Tree)
                Tree.append(child.data)
            else:
                Tree.append(child.data)
