# Here we use the library network x to inpute all of the data that was initially injected are introduced in a knowledge graph where all the labels are topologically inserted

# === build_graph_final.py === 4th step
# Final, working graph builder with GraphML-safe attributes
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter
import os
from datetime import datetime

print("🚀 Starting graph construction...")

# ———————————————————————
# 1. LOAD DATA (Assumes data is already in memory)
# ———————————————————————
# You should have already run:
#   amazon_df_cleaned, international_df_cleaned, sale_df_cleaned

if 'amazon_df_cleaned' not in globals():
    raise RuntimeError("❌ amazon_df_cleaned not found. Run data loading first.")
if 'international_df_cleaned' not in globals():
    raise RuntimeError("❌ international_df_cleaned not found. Run data loading first.")
if 'sale_df_cleaned' not in globals():
    raise RuntimeError("❌ sale_df_cleaned not found. Run data loading first.")

print("✅ All DataFrames are available.")

# Convert date columns to datetime
for df in [amazon_df_cleaned, international_df_cleaned]:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        raise ValueError("❌ 'date' column missing in data")

# ———————————————————————
# 2. CREATE DIRECTED GRAPH
# ———————————————————————
G = nx.DiGraph()
print("🏗️ Creating directed graph...")

# ———————————————————————
# 3. ADD INTERNATIONAL DATA
# ———————————————————————
if not international_df_cleaned.empty:
    print("📥 Adding International data...")
    for index, row in international_df_cleaned.iterrows():
        customer_id = f"intl_cust_{row['customer']}"
        product_sku = row['sku']
        order_id = f"intl_order_{index}"

        # Add Customer
        if customer_id not in G:
            G.add_node(customer_id, label='Customer', name=row['customer'], source='international')

        # Add Product
        if product_sku not in G:
            G.add_node(product_sku, label='Product', sku=product_sku)

        # Add Order
        if order_id not in G:
            G.add_node(order_id, label='Order', id=order_id,
                       date=str(row['date']), amount=row['gross_amt'], source='international')

        # Add Edges
        G.add_edge(customer_id, order_id, type='PURCHASED', timestamp=str(row['date']), dvid=1)
        G.add_edge(order_id, product_sku, type='CONTAINS', pcs=row['pcs'], rate=row['rate'],
                   timestamp=str(row['date']), dvid=1)

# ———————————————————————
# 4. ADD AMAZON DATA
# ———————————————————————
if not amazon_df_cleaned.empty:
    print("📥 Adding Amazon data...")
    for index, row in amazon_df_cleaned.iterrows():
        customer_id = f"amazon_cust_{row['ship_city']}_{row['ship_postal_code']}"
        product_sku = row['sku']
        order_id = f"amazon_order_{row['order_id']}"
        location_id = f"loc_{row['ship_city']}_{row['ship_state']}_{row['ship_postal_code']}"

        # Add Customer
        if customer_id not in G:
            G.add_node(customer_id, label='Customer',
                       city=row['ship_city'], state=row['ship_state'],
                       postal_code=row['ship_postal_code'], country=row['ship_country'],
                       source='amazon')

        # Add Product
        if product_sku not in G:
            G.add_node(product_sku, label='Product', sku=product_sku)

        # Add Order
        if order_id not in G:
            G.add_node(order_id, label='Order', id=order_id,
                       date=str(row['date']), status=row['status'],
                       service_level=row['ship_service_level'], amount=row['amount'],
                       source='amazon')

        # Add Location
        if location_id not in G:
            G.add_node(location_id, label='Location',
                       city=row['ship_city'], state=row['ship_state'],
                       postal_code=row['ship_postal_code'], country=row['ship_country'])

        # Add Category
        category_name = row['category']
        if pd.notna(category_name) and category_name not in G:
            G.add_node(category_name, label='Category', name=category_name)

        # Add Edges
        G.add_edge(customer_id, order_id, type='PURCHASED', timestamp=str(row['date']), dvid=1)
        G.add_edge(order_id, product_sku, type='CONTAINS', qty=row['qty'], timestamp=str(row['date']), dvid=1)
        G.add_edge(order_id, location_id, type='SHIPPED_TO', timestamp=str(row['date']), dvid=1)
        if pd.notna(category_name):
            if not G.has_edge(product_sku, category_name):
                G.add_edge(product_sku, category_name, type='BELONGS_TO', dvid=1)

# ———————————————————————
# 5. ENRICH PRODUCTS FROM SALE REPORT
# ———————————————————————
if not sale_df_cleaned.empty:
    print("📥 Enriching products from Sale Report data...")

    # Standardize and rename columns
    sale_df_cleaned.columns = [col.strip().replace(" ", "_").lower() for col in sale_df_cleaned.columns]

    # Map original names to clean ones
    column_map = {
        'SKU Code': 'sku_code',
        'Design No.': 'design_no.',
        'Stock': 'stock',
        'Category': 'category',
        'Size': 'size',
        'Color': 'color'
    }

    # Rename columns
    sale_df_cleaned.rename(columns=column_map, inplace=True)
    print(f"✅ Columns after renaming: {list(sale_df_cleaned.columns)}")

    # Now safe to use
    for index, row in sale_df_cleaned.iterrows():
        product_sku = str(row['sku_code']).strip()
        category_name = row['category']
        size = row['size']
        color = row['color']
        stock = row['stock']

        if product_sku in G:
            G.nodes[product_sku].update({
                'category': category_name,
                'size': size,
                'color': color,
                'stock': stock,
                'design_no': row['design_no.'] if 'design_no.' in row else None
            })
        else:
            G.add_node(product_sku, label='Product', sku=product_sku,
                       category=category_name, size=size, color=color,
                       stock=stock, design_no=row['design_no.'] if 'design_no.' in row else None)

        if pd.notna(category_name) and category_name not in G:
            G.add_node(category_name, label='Category', name=category_name)

        if pd.notna(category_name) and not G.has_edge(product_sku, category_name):
            G.add_edge(product_sku, category_name, type='BELONGS_TO', dvid=1)

# ———————————————————————
# 6. MAKE GRAPHML-SAFE (Fix Timestamps, np types, etc.)
# ———————————————————————
def make_graphml_safe(G):
    """Convert all attributes to GraphML-safe types: str, int, float, bool, None"""
    print("🔧 Converting attributes to GraphML-safe types...")

    for node, data in G.nodes(data=True):
        for key, value in list(data.items()):
            if pd.isna(value):
                data[key] = None
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                data[key] = value.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(value) else None
            elif isinstance(value, (np.integer, np.int64)):
                data[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                data[key] = float(value)
            elif isinstance(value, (list, tuple, dict)):
                data[key] = str(value)
            elif not isinstance(value, (str, int, float, bool, type(None))):
                data[key] = str(value)

    for u, v, data in G.edges(data=True):
        for key, value in list(data.items()):
            if pd.isna(value):
                data[key] = None
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                data[key] = value.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(value) else None
            elif isinstance(value, (np.integer, np.int64)):
                data[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                data[key] = float(value)
            elif isinstance(value, (list, tuple, dict)):
                data[key] = str(value)
            elif not isinstance(value, (str, int, float, bool, type(None))):
                data[key] = str(value)

    return G

G_safe = make_graphml_safe(G)

# ———————————————————————
# 7. VERIFY GRAPH
# ———————————————————————
print("\n✅ GRAPH CREATION COMPLETE")
print(f"   Nodes: {G_safe.number_of_nodes()}")
print(f"   Edges: {G_safe.number_of_edges()}")

# Node labels
node_labels = Counter(nx.get_node_attributes(G_safe, 'label').values())
print("\nNode Labels:")
for label, count in node_labels.items():
    print(f"  {label}: {count}")

# Edge types
edge_types = Counter(nx.get_edge_attributes(G_safe, 'type').values())
print("\nEdge Types:")
for etype, count in edge_types.items():
    print(f"  {etype}: {count}")

# Example nodes/edges
print("\n🔍 Example Nodes:")
for i, (node, data) in enumerate(G_safe.nodes(data=True)):
    if i >= 3: break
    print(f"  {node} -> {data}")

print("\n🔍 Example Edges:")
for i, (u, v, data) in enumerate(G_safe.edges(data=True)):
    if i >= 3: break
    print(f"  {u} -> {v} | {data}")

# ———————————————————————
# 8. SAVE GRAPH
# ———————————————————————
os.makedirs('graphs', exist_ok=True)
nx.write_graphml(G_safe, 'graphs/graph.graphml')
print("✅ Graph saved to: graphs/graph.graphml")

# Optional: Save with joblib (more flexible)
import joblib
joblib.dump(G_safe, 'graphs/graph.pkl')
print("✅ Graph also saved as pickle: graphs/graph.pkl")

print("\n🎉 Graph building complete. Ready for Node2Vec and link prediction.")
