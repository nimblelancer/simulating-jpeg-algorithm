class Node:
    def __init__(self, value, freq):
        self.value = value 
        self.freq = freq
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        return self.freq < other.freq