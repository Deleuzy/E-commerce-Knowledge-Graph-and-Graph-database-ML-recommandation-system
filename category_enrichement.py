# Add category properties to the items in the knowledge graphs

# === category_enrich.py === 10th step
# Add category-based features to graph and customer segments
import networkx as nx
import pandas as pd
import joblib

print("🔧 Enriching graph with category-aware features...")

# Load the UPDATED graph:
try:
    G = joblib.load('graphs/graph_enriched.pkl')  # ← Load updated graph
    print(f"✅ Loaded updated graph with {G.number_of_nodes():,} nodes")
except FileNotFoundError:
    try:
        G = joblib.load('graphs/graph_enriched_dvid2.pkl')  # ← Fallback
        print(f"✅ Loaded DVID-2 graph with {G.number_of_nodes():,} nodes")
    except FileNotFoundError:
        try:
            G = joblib.load('graphs/graph.pkl')
            print(f"⚠️  Loaded old graph with {G.number_of_nodes():,} nodes")
        except FileNotFoundError:
            raise FileNotFoundError("❌ No graph found. Run graph injection first.")

# Normalize categories
print("🧹 Normalizing product categories...")
for node_id, data in G.nodes(data=True):
    if data.get('label') == 'Product':
        cat = data.get('category')
        if pd.isna(cat):
            G.nodes[node_id]['category'] = 'Unknown'
        else:
            G.nodes[node_id]['category'] = str(cat).strip().lower()

# Update Louvain communities with category-based segment logic
try:
    from community import community_louvain
    G_similarity = joblib.load('models/G_similarity.pkl')
    partition = community_louvain.best_partition(G_similarity, weight='similarity')

    print("🏷️  Assigning category-based segments...")
    for node_id, comm_id in partition.items():
        if node_id in G and G.nodes[node_id].get('label') == 'Customer':
            # Get most frequent category
            cats = []
            for p in G.neighbors(node_id):
                if G.nodes[p].get('label') == 'Product':
                    cats.append(G.nodes[p].get('category', 'unknown'))
            if cats:
                from collections import Counter
                most_common_cat = Counter(cats).most_common(1)[0][0].title()
                G.nodes[node_id]['preferred_category'] = most_common_cat
            else:
                G.nodes[node_id]['preferred_category'] = 'Unknown'

            G.nodes[node_id]['segment'] = f'Segment-{comm_id}'
            G.nodes[node_id]['communityId'] = int(comm_id)
except:
    print("⚠️ Could not enrich with Louvain — using basic community logic")

# Save updated graph
joblib.dump(G, 'graphs/graph_enriched.pkl')
print("✅ Graph enriched and saved.")
