import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
import heapq
from io import BytesIO
import pickle

class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    # Create a priority queue (min heap) from the frequencies
    heap = [HuffmanNode(value, freq) for value, freq in frequencies.items()]
    heapq.heapify(heap)
    
    # Build the Huffman tree by merging nodes
    while len(heap) > 1:
        # Extract the two nodes with lowest frequencies
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # Create a new internal node with these two nodes as children
        # and with frequency equal to the sum of the two nodes' frequencies
        internal = HuffmanNode(None, left.freq + right.freq)
        internal.left = left
        internal.right = right
        
        # Add the internal node back to the priority queue
        heapq.heappush(heap, internal)
    
    # Return the root of the Huffman tree
    return heap[0] if heap else None

def generate_huffman_codes(node, code="", codes=None):
    if codes is None:
        codes = {}
    
    # If this is a leaf node, assign the current code to the value
    if node.value is not None:
        codes[node.value] = code
    else:
        # Traverse left (add 0)
        if node.left:
            generate_huffman_codes(node.left, code + "0", codes)
        # Traverse right (add 1)
        if node.right:
            generate_huffman_codes(node.right, code + "1", codes)
    
    return codes

def visualize_huffman_tree(node, ax, x=0, y=0, dx=1, level=0, pos=None):
    if pos is None:
        pos = {}
    
    # Store the position of this node
    pos[id(node)] = (x, y)
    
    # Draw node
    circle = plt.Circle((x, y), 0.2, fill=True, 
                        color='skyblue' if node.value is None else 'lightgreen', 
                        alpha=0.7)
    ax.add_patch(circle)
    
    # Add text (value and frequency)
    if node.value is not None:
        # It's a leaf node, show the value
        if isinstance(node.value, tuple):
              # For RGB tuples
            label = str(node.value)
        else:
            label = f"{node.value}"
    else:
        # Internal node, just show frequency
        label = f"{node.freq}"
    
    ax.text(x, y, label, ha='center', va='center', fontsize=8)

    # Draw children and connecting lines
    if node.left:
        new_x = x - dx / (2 ** level)
        new_y = y - 1
        ax.plot([x, new_x], [y, new_y], color='black')
        visualize_huffman_tree(node.left, ax, new_x, new_y, dx, level + 1, pos)
    
    if node.right:
        new_x = x + dx / (2 ** level)
        new_y = y - 1
        ax.plot([x, new_x], [y, new_y], color='black')
        visualize_huffman_tree(node.right, ax, new_x, new_y, dx, level + 1, pos)