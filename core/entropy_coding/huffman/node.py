class Node:
    """
    Lớp biểu diễn node trong cây Huffman.
    
    Attributes:
    -----------
    value : any
        Giá trị của node (None nếu là node cha)
    freq : int
        Tần suất xuất hiện của giá trị
    left : Node
        Node con trái
    right : Node
        Node con phải
    """
    def __init__(self, value, freq):
        if freq < 0:
            raise ValueError("Tần suất phải không âm")
        self.value = value 
        self.freq = freq
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        return self.freq < other.freq