# Here we analyze the centrality degrees inside of the graph for the different users and items to further the feature engineering

# === analyze_degrees.py ===
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import os
import joblib
import numpy as np
from scipy.stats import percentileofscore

print("📊 Starting COMPREHENSIVE Degree Analysis + Feature Engineering...")

# Load graph
try:
    G = joblib.load('graphs/graph_enriched.pkl')
    print(f"✅ Loaded graph with {G.number_of_nodes():,} nodes")
except FileNotFoundError:
    raise FileNotFoundError("❌ No graph found.")

os.makedirs("analysis", exist_ok=True)

# Get all degrees
all_degrees = dict(G.degree())

# ———————————————————————
# 1. ADD TRANSFORMED DEGREE FEATURES
# ———————————————————————
print("🧮 Calculating transformed degree features...")

# Log degree
for node in G.nodes():
    degree = G.degree(node)
    G.nodes[node]['log_degree'] = np.log1p(degree)  # log(1+x) to handle 0

# Percentile rank (global)
all_deg_list = list(all_degrees.values())
for node in G.nodes():
    degree = G.degree(node)
    G.nodes[node]['degree_percentile'] = percentileofscore(all_deg_list, degree) / 100.0

# Z-score (global)
mean_deg = np.mean(all_deg_list)
std_deg = np.std(all_deg_list)
for node in G.nodes():
    degree = G.degree(node)
    G.nodes[node]['degree_zscore'] = (degree - mean_deg) / std_deg if std_deg > 0 else 0.0

# ———————————————————————
# 2. NODE-TYPE SPECIFIC NORMALIZATION
# ———————————————————————
print("🧍🛍️  Calculating node-type-specific degree features...")

node_types = ['Customer', 'Product', 'Order']
type_degrees = {nt: [] for nt in node_types}

# Collect degrees by type
for node, data in G.nodes(data=True):
    node_type = data.get('label')
    if node_type in node_types:
        type_degrees[node_type].append(G.degree(node))

# Add normalized features
for node, data in G.nodes(data=True):
    node_type = data.get('label')
    degree = G.degree(node)
    if node_type in type_degrees and len(type_degrees[node_type]) > 1:
        type_mean = np.mean(type_degrees[node_type])
        type_std = np.std(type_degrees[node_type])
        G.nodes[node]['type_normalized_degree'] = (degree - type_mean) / type_std if type_std > 0 else 0.0
    else:
        G.nodes[node]['type_normalized_degree'] = 0.0

# ———————————————————————
# 3. COMMUNITY-AWARE DEGREE FEATURES — WITH PARTITION FALLBACK
# ———————————————————————
print("👥 Calculating community-aware degree features...")

# 🆕 LOAD PARTITION AS FALLBACK
partition = None
try:
    partition = joblib.load('models/customer_partition.pkl')
    print("✅ Loaded community partition from models/customer_partition.pkl")
except FileNotFoundError:
    print("⚠️  customer_partition.pkl not found — will rely on graph attributes")

# Check if ANY community info exists
has_community_in_graph = any('communityId' in data for _, data in G.nodes(data=True))
has_partition = partition is not None

if has_community_in_graph or has_partition:
    community_degrees = {}

    # Build community_degrees using graph OR partition
    for node in G.nodes():
        comm = -1
        # Priority 1: get from graph
        if 'communityId' in G.nodes[node]:
            comm = G.nodes[node]['communityId']
        # Fallback: get from partition
        elif has_partition and node in partition:
            comm = partition[node]

        if comm != -1:
            if comm not in community_degrees:
                community_degrees[comm] = []
            community_degrees[comm].append(G.degree(node))

    # Add features to nodes
    for node in G.nodes():
        comm = -1
        if 'communityId' in G.nodes[node]:
            comm = G.nodes[node]['communityId']
        elif has_partition and node in partition:
            comm = partition[node]

        degree = G.degree(node)
        if comm in community_degrees and len(community_degrees[comm]) > 1:
            comm_mean = np.mean(community_degrees[comm])
            comm_std = np.std(community_degrees[comm])
            G.nodes[node]['comm_normalized_degree'] = (degree - comm_mean) / comm_std if comm_std > 0 else 0.0
            G.nodes[node]['comm_degree_percentile'] = percentileofscore(community_degrees[comm], degree) / 100.0
        else:
            G.nodes[node]['comm_normalized_degree'] = 0.0
            G.nodes[node]['comm_degree_percentile'] = 0.0

    print("✅ Community-aware features added successfully!")
else:
    print("❌ No community info found in graph or partition — skipping community features")

# ———————————————————————
# 4. SAVE ENRICHED GRAPH
# ———————————————————————
joblib.dump(G, 'graphs/graph_enriched_with_degree_features.pkl')
print("✅ Saved graph with enriched degree features: graphs/graph_enriched_with_degree_features.pkl")

# ———————————————————————
# 5. SANITY CHECK — VERIFY FEATURES WERE ADDED
# ———————————————————————
sample_node = next(iter(G.nodes()))
print("\n🔍 Sanity Check — Sample Node Features:")
print(f"   log_degree: {G.nodes[sample_node].get('log_degree', 'MISSING')}")
print(f"   degree_percentile: {G.nodes[sample_node].get('degree_percentile', 'MISSING')}")
print(f"   comm_degree_percentile: {G.nodes[sample_node].get('comm_degree_percentile', 'MISSING')}")

if 'comm_degree_percentile' in G.nodes[sample_node]:
    print("🎉 SUCCESS: Community-aware features are present!")
else:
    print("⚠️  WARNING: Community-aware features may be missing — double-check Louvain ran and saved graph.")

# ———————————————————————
# 6. ANALYSIS & PLOTS (UNCHANGED)
# ———————————————————————
customer_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'Customer']
customer_degrees = [G.degree(n) for n in customer_nodes]

plt.figure(figsize=(10, 6))
plt.hist(customer_degrees, bins=range(1, max(customer_degrees)+2), alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Customer Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Count")
plt.grid(axis='y', alpha=0.3)
plt.savefig("analysis/customer_degree_distribution.png", dpi=120, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("✅ DEGREE FEATURE ENGINEERING COMPLETE")
print("   Added features: log_degree, degree_percentile, degree_zscore, type_normalized_degree")
if has_community_in_graph or has_partition:
    print("   Added community features: comm_normalized_degree, comm_degree_percentile")
print("   Output graph saved with all new features")
print("="*60)
