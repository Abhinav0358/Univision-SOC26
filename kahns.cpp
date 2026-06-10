#include <iostream>
#include <vector>
#include <queue>

// Performs topological sort using Kahn's Algorithm.
// Returns a vector with the sorted order, or an empty vector if a cycle is detected.
std::vector<int> topologicalSort(int numNodes, const std::vector<std::pair<int, int>>& edges) {
    // 1. Initialize adjacency list and in-degree counter
    std::vector<std::vector<int>> adj(numNodes);
    std::vector<int> inDegree(numNodes, 0);

    // Build the graph and calculate in-degrees
    // edges[i].first is the parent node, edges[i].second is the child node
    for (const auto& edge : edges) {
        int u = edge.first;
        int v = edge.second;
        adj[u].push_back(v); // Track children
        inDegree[v]++;       // Track parents (incoming edges)
    }

    // 2. Queue to track nodes with no dependencies (in-degree of 0)
    std::queue<int> q;
    for (int i = 0; i < numNodes; ++i) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }

    // List to store the final execution order
    std::vector<int> order;

    // 3. Process the nodes
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        order.push_back(u);

        // Decrease the in-degree of all children
        for (int v : adj[u]) {
            inDegree[v]--;
            // If child has no remaining parent dependencies, add to queue
            if (inDegree[v] == 0) {
                q.push(v);
            }
        }
    }

    // 4. Cycle Detection
    // If we couldn't process all nodes, there is a cycle (the graph is not a DAG)
    if (order.size() != static_cast<size_t>(numNodes)) {
        return {}; // Return empty vector indicating failure
    }

    return order;
}

int main() {
    // Example Workflow (DAG):
    // 0: Load Image
    // 1: Resize Image (depends on 0)
    // 2: Grayscale Conversion (depends on 1)
    // 3: Object Detection (depends on 1)
    // 4: Display Results (depends on 2 and 3)

    int numNodes = 5;
    std::vector<std::pair<int, int>> edges = {
        {0, 1}, // 0 -> 1
        {1, 2}, // 1 -> 2
        {1, 3}, // 1 -> 3
        {2, 4}, // 2 -> 4
        {3, 4}  // 3 -> 4
    };

    std::vector<int> executionOrder = topologicalSort(numNodes, edges);

    if (executionOrder.empty()) {
        std::cout << "Error: Cycle detected! Workflow cannot be executed." << std::endl;
    } else {
        std::cout << "Valid execution order determined: ";
        for (int node : executionOrder) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
