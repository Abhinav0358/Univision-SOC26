"""
Big Data Processing Foundations

Introduces distributed computing paradigms including MapReduce, 
Hadoop storage and resource management, Apache Spark memory optimizations, 
and Cloud Computing models.
"""

# 1. MapReduce Paradigm
# Designed for parallel data processing across distributed clusters:
# - Map: Worker nodes process input data partitions, emitting intermediate (Key, Value) pairs.
# - Shuffle & Sort: Collects and groups identical intermediate keys together.
# - Reduce: Worker nodes aggregate grouped values, producing final results.

# Conceptual example: Word Count
docs = ["big data is big", "data is power"]

# Mapping phase:
# doc 1 -> ("big", 1), ("data", 1), ("is", 1), ("big", 1)
# doc 2 -> ("data", 1), ("is", 1), ("power", 1)

# Shuffling phase:
# "big" -> [1, 1]
# "data" -> [1, 1]
# "is" -> [1, 1]
# "power" -> [1]

# Reducing phase:
# "big": 2, "data": 2, "is": 2, "power": 1

# 2. Hadoop Ecosystem Architecture
# Core components of Apache Hadoop:
# - HDFS (Hadoop Distributed File System): Splits files into block sequences and distributes/replicates
#   them across cluster nodes for fault tolerance.
# - YARN (Yet Another Resource Negotiator): Coordinates cluster resource allocation and schedules applications.
# - MapReduce: Distributed execution framework.

# 3. Apache Spark Optimization
# Enhances processing speed relative to MapReduce by keeping intermediate computations in RAM:
# - RDD (Resilient Distributed Dataset): Fault-tolerant collection of elements partitioned across cluster nodes.
# - Lazy Evaluation: Spark builds an execution plan represented as a Directed Acyclic Graph (DAG),
#   optimizing performance before executing any transformations.

print("--- Big Data Framework Comparison ---")
print("Hadoop: Disk-based, batch-oriented, reliable storage storage.")
print("Spark: Memory-centric, batch and stream processing, optimized DAG execution.")

# 4. Cloud Computing Models
# - IaaS (Infrastructure as a Service): Renting raw compute, storage, and networking resources (e.g., AWS EC2).
# - PaaS (Platform as a Service): Renting runtime environments to deploy applications without managing underlying OS (e.g., Google App Engine).
# - SaaS (Software as a Service): Utilizing end-user software over the internet (e.g., Gmail, Salesforce).

print("\n[SUCCESS] Big data foundations lab completed.")
print("Revision of all 12 Data Science labs is complete.")
