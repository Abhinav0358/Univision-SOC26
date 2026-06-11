"""
Lab 12: Big Data Foundations - Deep Dive Tutorial
================================================
Syllabus Topics: MapReduce, Hadoop (HDFS/YARN), Apache Spark (RDDs, DAGs).

This script explains how we process Petabytes of data by splitting
the work across thousands of computers.
"""

# =============================================================================
# 1. THE PARADIGM: MAPREDUCE
# =============================================================================
# Problem: One computer is too slow for big data.
# Solution: Divide and Conquer.
# - MAP: Split the data and give a piece to every computer. Each produces (Key, Value) pairs.
# - SHUFFLE/SORT: Group all identical keys together.
# - REDUCE: Aggregate the values for each key (e.g., Sum them up).

# Example: Word Count
docs = ["big data is big", "data is power"]

# 1. MAP:
#   doc 1 -> ("big", 1), ("data", 1), ("is", 1), ("big", 1)
#   doc 2 -> ("data", 1), ("is", 1), ("power", 1)

# 2. SHUFFLE:
#   "big" -> [1, 1]
#   "data" -> [1, 1]
#   "is" -> [1, 1]
#   "power" -> [1]

# 3. REDUCE:
#   "big": 2, "data": 2, "is": 2, "power": 1

# =============================================================================
# 2. THE ARCHITECTURE: HADOOP
# =============================================================================
# Hadoop is the "Operating System" for Big Data.
# - HDFS (Hadoop Distributed File System): How data is stored.
#   It splits files into blocks and replicates them across the cluster (for safety).
# - YARN (Resource Manager): The "Traffic Cop" that decides which computer
#   gets which task.
# - MAPREDUCE (v1): The engine that processes the data.

# =============================================================================
# 3. THE EVOLUTION: APACHE SPARK
# =============================================================================
# MapReduce was slow because it wrote to the DISK after every step.
# Spark is fast because it keeps data in RAM (IN-MEMORY).

# - RDD (Resilient Distributed Dataset): The fundamental data object in Spark.
#   - Resilient: If a computer fails, Spark can rebuild the data.
#   - Distributed: Split across the cluster.
# - DAG (Directed Acyclic Graph): Spark's execution plan.
#   It doesn't run anything until you ask for a result (LAZY EVALUATION).
#   This allows Spark to optimize the whole path.

print("--- Big Data Summary ---")
print("Hadoop: Disk-based, Batch-oriented, Reliable.")
print("Spark: Memory-based, Real-time & Batch, Very Fast.")

# =============================================================================
# 4. CLOUD COMPUTING
# =============================================================================
# - IaaS (Infrastructure): You rent the hardware (AWS EC2).
# - PaaS (Platform): You rent the environment (Google App Engine).
# - SaaS (Software): You use the software (Gmail, Salesforce).

print("\n[SUCCESS] Lab 12 Complete. You have finished the Data Science Revision!")
print("Go through these 12 files one by one to refresh your mastery.")
