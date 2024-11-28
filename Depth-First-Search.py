def dfs_recursive(graph, node, visited=None, dfs_order=None):
    """
    Perform Depth-First Search on a graph using recursion.

    :param graph: A dictionary where keys are nodes and values are lists of neighboring nodes.
    :param node: The current node being visited.
    :param visited: A set to track visited nodes (default is None).
    :param dfs_order: A list to store the DFS traversal order (default is None).
    :return: A list of nodes visited in DFS order.
    """
    if visited is None:
        visited = set()
    if dfs_order is None:
        dfs_order = []

    visited.add(node)  # Mark the current node as visited
    dfs_order.append(node)  # Add it to the DFS order

    for neighbor in graph[node]:  # Recur for all unvisited neighbors
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, dfs_order)

    return dfs_order


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
    print("DFS Traversal Order (Recursive):", dfs_recursive(graph, start_node))
