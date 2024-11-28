from collections import deque

def bfs(graph, start):
    """
    Perform Breadth-First Search on a graph.

    :param graph: A dictionary where keys are nodes and values are lists of neighboring nodes.
    :param start: The starting node for BFS.
    :return: A list of nodes visited in BFS order.
    """
    visited = set()  # To keep track of visited nodes
    queue = deque([start])  # Initialize a queue with the start node
    bfs_order = []  # List to store the BFS traversal order

    while queue:
        node = queue.popleft()  # Dequeue the next node
        if node not in visited:
            visited.add(node)  # Mark the node as visited
            bfs_order.append(node)  # Add it to the BFS order
            # Enqueue all unvisited neighbors
            queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)

    return bfs_order


# Example Usage
if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    start_node = 'A'
    print("BFS Traversal Order:", bfs(graph, start_node))
