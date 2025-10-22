# Here we inject further data into the knowldge graphs once the first feature engineering and then the prediction algorithms and recommandations are done

# === dvid_graph_extension_only.py ===
# Clean DVID graph extension - dates don't matter with DVID tagging
import pandas as pd
import joblib
from datetime import datetime
import random

print("🔌 DVID GRAPH EXTENSION (Dates don't matter with DVID tagging)")
print("="*60)

# Configuration
CURRENT_DVID = 2
MAX_SAMPLES = 5000  # Safety limit

# Load existing graph
G = joblib.load('graphs/graph_enriched.pkl')
nodes_before = G.number_of_nodes()
edges_before = G.number_of_edges()
print(f"📊 Starting graph: {nodes_before:,} nodes, {edges_before:,} edges")

# Better date parsing
def safe_date_parse(date_str):
    """Robust date parsing for American formats"""
    if pd.isna(date_str):
        return datetime(2022, 6, 1)  # Default middle date

    formats = ['%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y', '%Y-%m-%d']
    date_str = str(date_str).strip()

    for fmt in formats:
        try:
            return datetime.strptime(date_str.split()[0] if ' ' in date_str else date_str, fmt)
        except:
            continue
    return datetime(2022, 6, 1)

# Load and process data
print(f"\n📥 Loading data...")

try:
    # Amazon data
    amazon_df = pd.read_csv('Amazon-Sale-Report.csv', low_memory=False).dropna()
    amazon_df.columns = [col.strip().replace("-", "_").replace(" ", "_").lower() for col in amazon_df.columns]
    amazon_df['parsed_date'] = amazon_df['date'].apply(safe_date_parse)
    print(f"📊 Amazon: {len(amazon_df):,} records")

    # International data
    intl_df = pd.read_csv('International-sale-Report.csv', low_memory=False).dropna()
    intl_df.columns = [col.strip().replace("-", "_").replace(" ", "_").lower() for col in intl_df.columns]
    intl_df['customer'] = intl_df['customer'].astype(str).str.lower()

    # Filter out month names
    month_names = ['aug-21', 'dec-21', 'feb-22', 'jan-22', 'jul-21', 'mar-22', 'nov-21', 'oct-21', 'sep-21', 'jun-21']
    intl_df = intl_df[~intl_df['customer'].isin(month_names)]
    intl_df['parsed_date'] = intl_df['date'].apply(safe_date_parse)
    print(f"📊 International: {len(intl_df):,} records")

except Exception as e:
    print(f"❌ Data loading failed: {e}")
    exit()

# Sample if too large
total_records = len(amazon_df) + len(intl_df)
if total_records > MAX_SAMPLES:
    print(f"⚖️ Sampling down to {MAX_SAMPLES:,} records...")
    amazon_df = amazon_df.sample(n=min(len(amazon_df), MAX_SAMPLES//2), random_state=42)
    intl_df = intl_df.sample(n=min(len(intl_df), MAX_SAMPLES//2), random_state=42)

# Add data to graph with DVID tagging (dates don't matter!)
print(f"\n🔗 Adding DVID-{CURRENT_DVID} data to graph...")

# Amazon data
for idx, row in amazon_df.iterrows():
    customer_id = f"amazon_customer_{idx}_dvid{CURRENT_DVID}"
    order_id = f"amazon_order_{idx}_dvid{CURRENT_DVID}"
    product_sku = row.get('sku', f'product_{idx}')
    order_date = row['parsed_date']

    # Add nodes with DVID tagging
    if customer_id not in G:
        G.add_node(customer_id, label='Customer', source='amazon', dvid=CURRENT_DVID)
    if product_sku not in G:
        G.add_node(product_sku, label='Product', sku=product_sku, dvid=CURRENT_DVID)
    if order_id not in G:
        G.add_node(order_id, label='Order', date=order_date, dvid=CURRENT_DVID)

    # Add edges with DVID tagging
    G.add_edge(customer_id, order_id, type='PURCHASED', timestamp=order_date, dvid=CURRENT_DVID)
    G.add_edge(order_id, product_sku, type='CONTAINS', qty=row.get('qty', 1), timestamp=order_date, dvid=CURRENT_DVID)

# International data
for idx, row in intl_df.iterrows():
    customer_id = f"intl_customer_{idx}_dvid{CURRENT_DVID}"
    order_id = f"intl_order_{idx}_dvid{CURRENT_DVID}"
    product_sku = row.get('sku', f'product_{idx}')
    order_date = row['parsed_date']

    # Add nodes with DVID tagging
    if customer_id not in G:
        G.add_node(customer_id, label='Customer', source='international', dvid=CURRENT_DVID)
    if product_sku not in G:
        G.add_node(product_sku, label='Product', sku=product_sku, dvid=CURRENT_DVID)
    if order_id not in G:
        G.add_node(order_id, label='Order', date=order_date, dvid=CURRENT_DVID)

    # Add edges with DVID tagging
    G.add_edge(customer_id, order_id, type='PURCHASED', timestamp=order_date, dvid=CURRENT_DVID)
    G.add_edge(order_id, product_sku, type='CONTAINS', qty=row.get('qty', 1), timestamp=order_date, dvid=CURRENT_DVID)

# Summary
nodes_after = G.number_of_nodes()
edges_after = G.number_of_edges()

print(f"\n📊 DVID-{CURRENT_DVID} Extension Complete:")
print(f"   Nodes: {nodes_before:,} → {nodes_after:,} (+{nodes_after - nodes_before:,})")
print(f"   Edges: {edges_before:,} → {edges_after:,} (+{edges_after - edges_before:,})")

# Verify DVID tagging
dvid_nodes = sum(1 for n, data in G.nodes(data=True) if data.get('dvid') == CURRENT_DVID)
dvid_edges = sum(1 for u, v, data in G.edges(data=True) if data.get('dvid') == CURRENT_DVID)
print(f"   DVID-{CURRENT_DVID} nodes: {dvid_nodes:,}")
print(f"   DVID-{CURRENT_DVID} edges: {dvid_edges:,}")

# Save graph
print(f"\n💾 Saving graph...")
joblib.dump(G, 'graphs/graph_enriched.pkl')
joblib.dump(G, f'graphs/graph_enriched_dvid{CURRENT_DVID}.pkl')
