#include <iostream>
#include <vector>
#include <queue>

// Computes a topological ordering of a DAG using Kahn's algorithm.
// Returns the sorted node indices, or an empty vector if a cycle is detected.
std::vector<int> topologicalSort(int numNodes, const std::vector<std::pair<int, int>>& edges) {
    std::vector<std::vector<int>> adj(numNodes);
    std::vector<int> inDegree(numNodes, 0);

    // Build the adjacency list and compute incoming degrees for each node
    for (const auto& edge : edges) {
        int u = edge.first;
        int v = edge.second;
        adj[u].push_back(v);
        inDegree[v]++;
    }

    // Queue nodes that have no incoming dependencies (in-degree == 0)
    std::queue<int> q;
    for (int i = 0; i < numNodes; ++i) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }

    std::vector<int> order;

    // Process nodes layer by layer, removing edges as we go
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        order.push_back(u);

        for (int v : adj[u]) {
            inDegree[v]--;
            // Node v is ready to be processed once all its dependencies are resolved
            if (inDegree[v] == 0) {
                q.push(v);
            }
        }
    }

    // If we couldn't process all nodes, the graph contains a cycle
    if (order.size() != static_cast<size_t>(numNodes)) {
        return {};
    }

    return order;
}

int main() {
    // Setup a sample DAG representing a simple vision pipeline dependency graph:
    // 0 (Load) -> 1 (Resize) -> 2 (Grayscale) -> 4 (Display)
    // 1 (Resize) -> 3 (Object Detection) -> 4 (Display)

    int numNodes = 5;
    std::vector<std::pair<int, int>> edges = {
        {0, 1},
        {1, 2},
        {1, 3},
        {2, 4},
        {3, 4}
    };

    std::vector<int> executionOrder = topologicalSort(numNodes, edges);

    if (executionOrder.empty()) {
        std::cout << "Error: Cycle detected! The dependency graph is not a DAG." << std::endl;
    } else {
        std::cout << "Valid execution order: ";
        for (int node : executionOrder) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
