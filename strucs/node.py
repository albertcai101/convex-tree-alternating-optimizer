class Node:
    def __init__(self, depth, D, K,
                 parent, left, right,):
        self.depth = depth
        self.D = D
        self.K = K
        self.parent = parent
        self.left = left
        self.right = right

    def set_children(self, left, right):
        self.left = left
        self.right = right

    def set_parent(self, node):
        self.parent = node

class StandardNode(Node):
    def __init__(self, depth, D, K,
                 parent, left, right,
                 w, b,):
        super().__init__(depth, D, K, parent, left, right)
        self.w = w
        self.b = b

    @property
    def is_leaf(self):
        return False
    
    def copy(self):
        return StandardNode(self.depth, self.D, self.K, 
                            self.parent, self.left, self.right, 
                            self.w, self.b)

class LeafNode(Node):
    def __init__(self, depth, D, K, 
                 parent,
                 label,):
        super().__init__(depth=depth, D=D, K=K, parent=parent, left=None, right=None)
        self.label = label

    @property
    def is_leaf(self):
        return True
    
    def copy(self):
        return LeafNode(self.depth, self.D, self.K, 
                        self.parent,
                        self.label)