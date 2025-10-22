# === xgb_balanced_adam_sampling.py ===
# Balanced Adam-style sampling for XGBoost recommendations
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import xgboost as xgb
import random
import os

print("🎯 Starting: BALANCED Adam-Style XGBoost Recommendations")

# ———————————————————————
# 1. LOAD REQUIRED ARTIFACTS
# ———————————————————————
try:
    G = joblib.load('graphs/graph_enriched.pkl')
    print(f"✅ Loaded enriched graph with {G.number_of_nodes()} nodes")
except:
    try:
        G = joblib.load('graphs/graph.pkl')
        print(f"✅ Loaded basic graph with {G.number_of_nodes()} nodes")
    except:
        raise FileNotFoundError("❌ No graph found. Run graph building scripts first.")

try:
    customer_features_df = pd.read_csv('models/customer_features.csv')
    product_features_df = pd.read_csv('models/product_features.csv')
    print(f"✅ Loaded customer features: {len(customer_features_df)}")
    print(f"✅ Loaded product features: {len(product_features_df)}")
except:
    raise FileNotFoundError("❌ Run feature generation script first.")

# Load XGBoost model with discounts
try:
    xgb_model = xgb.Booster()
    xgb_model.load_model('models/xgb_robust_discount.json')
    print("✅ Loaded XGBoost model with discount features")

    class XGBWrapper:
        def predict_proba(self, X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            dmat = xgb.DMatrix(X)
            pred = self.model.predict(dmat)
            return np.column_stack([1 - pred, pred])

    xgb_wrapper = XGBWrapper()
    xgb_wrapper.model = xgb_model

except Exception as e:
    raise FileNotFoundError(f"❌ Could not load XGBoost discount model: {e}")

# ———————————————————————
# 2. BALANCED ADAM-STYLE SAMPLING
# ———————————————————————
def balanced_adam_sampling(probs, temperature=2.0, uniform_mixture=0.3, smoothing_factor=0.1):
    """
    Balanced Adam-style sampling:
    - Mixes uniform distribution with model predictions
    - Applies gentle temperature scaling
    - Adds smoothing to prevent extreme probability gaps
    """
    # Ensure we have valid probabilities
    probs = np.clip(probs, 1e-10, 1 - 1e-10)

    # Normalize original probabilities
    if np.sum(probs) > 0:
        normalized_model_probs = probs / np.sum(probs)
    else:
        normalized_model_probs = np.ones(len(probs)) / len(probs)

    # Create uniform distribution
    uniform_probs = np.ones(len(probs)) / len(probs)

    # Mix uniform and model probabilities (balanced approach)
    # uniform_mixture = 0.3 means 30% uniform, 70% model-based
    mixed_probs = uniform_mixture * uniform_probs + (1 - uniform_mixture) * normalized_model_probs

    # Apply gentle smoothing to reduce extreme differences
    smoothed_probs = mixed_probs + smoothing_factor
    smoothed_probs = smoothed_probs / np.sum(smoothed_probs)

    # Apply gentle temperature scaling
    logits = np.log(smoothed_probs)
    logits_scaled = logits / temperature

    # Apply softmax with numerical stability
    exp_logits = np.exp(logits_scaled - np.max(logits_scaled))
    sampling_probs = exp_logits / exp_logits.sum()

    # Final normalization and safety checks
    sampling_probs = np.nan_to_num(sampling_probs, nan=0.0)
    if sampling_probs.sum() > 0:
        sampling_probs = sampling_probs / sampling_probs.sum()
    else:
        sampling_probs = np.ones(len(probs)) / len(probs)

    return sampling_probs

# ———————————————————————
# 3. CONFIGURATION
# ———————————————————————
embedding_dim = 64
print(f"✅ Using embedding dimension: {embedding_dim}")

# ———————————————————————
# 4. TEMPORAL FEATURES
# ———————————————————————
def calculate_temporal_features(G, node, current_time):
    """Calculate temporal features"""
    timestamps = [
        data['timestamp'] for _, _, data in G.edges(node, data=True)
        if 'timestamp' in data and isinstance(data['timestamp'], datetime) and data['timestamp'] <= current_time
    ]
    if not timestamps:
        return {'recency': 999, 'frequency': 0}
    latest = max(timestamps)
    recency = (current_time - latest).days
    frequency = len(timestamps)
    return {'recency': recency, 'frequency': frequency}

# ———————————————————————
# 5. DYNAMIC DISCOUNT ASSIGNMENT
# ———————————————————————
def add_dynamic_discounts_with_tracking(G, simulated_date):
    """Apply seasonal discounts"""
    for node, data in G.nodes(data=True):
        if data.get('label') == 'Product':
            month = simulated_date.month
            if month == 12:  # December: big sale
                discount = round(random.uniform(0.40, 0.60), 2)
            elif month in [6, 7, 11]:
                discount = round(random.uniform(0.20, 0.35), 2)
            elif month in [1, 2, 8]:
                discount = round(random.uniform(0.10, 0.25), 2)
            else:
                discount = round(random.uniform(0.0, 0.10), 2)
            data['discount'] = discount

# ———————————————————————
# 6. FEATURE EXTRACTION
# ———————————————————————
def get_link_features_with_discount_compatible(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time):
    """Extract features compatible with trained model (134 features)"""
    try:
        c_row = cust_df[cust_df['node_id'] == cust_id].iloc[0]
        p_row = prod_df[prod_df['node_id'] == prod_id].iloc[0]
    except (IndexError, KeyError):
        return np.zeros(134)

    customer_degree = float(c_row.get('degree', 0))
    product_degree = float(p_row.get('degree', 0))

    try:
        c_emb = [float(c_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
        p_emb = [float(p_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
    except:
        c_emb = [0.0] * emb_dim
        p_emb = [0.0] * emb_dim

    c_temp = calculate_temporal_features(G, cust_id, current_time)
    p_temp = calculate_temporal_features(G, prod_id, current_time)
    current_discount = float(G.nodes[prod_id].get('discount', 0.0))

    features = np.array([
        customer_degree, product_degree,
        *c_emb, *p_emb,
        c_temp['recency'], c_temp['frequency'],
        p_temp['recency'], p_temp['frequency'],
        current_discount
    ])

    # Ensure exactly 134 features (remove the extra discount feature to match your model)
    if len(features) > 134:
        features = features[:134]
    elif len(features) < 134:
        features = np.pad(features, (0, 134 - len(features)), constant_values=0.0)

    return features

# ———————————————————————
# 7. GET VALID PRODUCTS
# ———————————————————————
def get_valid_products(G):
    """Get valid products for recommendation"""
    valid_products = []
    for node, data in G.nodes(data=True):
        if data.get('label') == 'Product':
            if (
                data.get('color') not in [None, 'N/A', 'Unknown', ''] and
                data.get('size') not in [None, 'N/A', 'Unknown', ''] and
                data.get('category') not in [None, 'N/A', 'Unknown', ''] and
                data.get('stock', 0) > 0
            ):
                valid_products.append(node)
    return valid_products

valid_products = get_valid_products(G)
print(f"✅ Filtered {len(valid_products)} valid products.")

# ———————————————————————
# 8. BALANCED ADAM-STYLE RECOMMENDATION ENGINE
# ———————————————————————
def get_recommendations_for_customer(G, customer_id, current_time, top_n=3):
    """Recommendation engine with balanced Adam-style sampling"""
    if customer_id not in G:
        return []

    # Get products customer hasn't bought
    purchased_products = set()
    for _, order_id, data in G.edges(customer_id, data=True):
        if data.get('type') == 'PURCHASED' and G.nodes[order_id].get('label') == 'Order':
            for _, product_id, order_data in G.edges(order_id, data=True):
                if order_data.get('type') == 'CONTAINS' and G.nodes[product_id].get('label') == 'Product':
                    purchased_products.add(product_id)

    potential_products = [p for p in valid_products if p not in purchased_products]

    if len(potential_products) == 0:
        return []

    # Sample products if too many
    if len(potential_products) > 1500:
        potential_products = random.sample(potential_products, 1500)
        print(f"🔍 Sampling {len(potential_products)} products")

    product_features = []
    for product_id in potential_products:
        link_features = get_link_features_with_discount_compatible(
            customer_id, product_id,
            customer_features_df, product_features_df,
            embedding_dim, current_time
        )
        product_features.append((product_id, link_features))

    if not product_features:
        return []

    product_ids, feature_vectors = zip(*product_features)
    feature_vectors = np.array(feature_vectors)

    # Predict probabilities
    try:
        probs = xgb_wrapper.predict_proba(feature_vectors)[:, 1]
        print(f"🔍 Raw probability range: {np.min(probs):.4f} to {np.max(probs):.4f}")
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return []

    # Apply balanced Adam-style sampling
    sampling_probs = balanced_adam_sampling(probs, temperature=2.0, uniform_mixture=0.3, smoothing_factor=0.05)
    print(f"🔍 Balanced Adam sampling applied (30% uniform, 70% model, temp=2.0)")

    # Sample with category diversity
    seen_categories = set()
    selected = []
    attempts = 0
    max_attempts = top_n * 150

    print(f"🔍 Sampling with {max_attempts} attempts")

    while len(selected) < top_n and attempts < max_attempts:
        if len(product_ids) == 0 or sampling_probs.sum() == 0:
            break

        try:
            idx = np.random.choice(len(product_ids), p=sampling_probs)
        except ValueError:
            idx = np.random.choice(len(product_ids))

        prod_id = product_ids[idx]
        prob = probs[idx]
        category = str(G.nodes[prod_id].get('category', 'Unknown')).upper().strip()

        # Stock check
        if G.nodes[prod_id].get('stock', 0) <= 0:
            attempts += 1
            continue

        # Category diversity (balanced approach)
        if category in seen_categories:
            # Allow some repetition but not too much
            category_count = sum(1 for s in selected if str(G.nodes[s[0]].get('category', 'Unknown')).upper().strip() == category)
            if category_count >= 2:  # Allow max 2 from same category
                attempts += 1
                continue

        selected.append((prod_id, prob))
        seen_categories.add(category)
        attempts += 1

    # Fallback if needed
    if len(selected) < top_n:
        print(f"⚠️ Fallback: Selecting diverse items")
        # Group by category and pick one from each category first
        category_groups = {}
        for i, prod_id in enumerate(product_ids):
            if G.nodes[prod_id].get('stock', 0) > 0:
                category = str(G.nodes[prod_id].get('category', 'Unknown')).upper().strip()
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append((prod_id, probs[i]))

        # Pick one item from each category
        for category, items in category_groups.items():
            if len(selected) >= top_n:
                break
            if category not in seen_categories:
                # Pick item with middle probability (not highest, not lowest)
                items.sort(key=lambda x: x[1])
                if len(items) > 2:
                    selected_item = items[len(items)//2]  # Pick middle item
                else:
                    selected_item = items[0]  # Pick first if few items
                selected.append(selected_item)
                seen_categories.add(category)

    return selected

# ———————————————————————
# 9. ENRICH RECOMMENDATIONS
# ———————————————————————
def enrich_recommendations(G, recommendations):
    """Add human-readable information"""
    enriched_recs = []
    for prod_id, prob in recommendations:
        data = G.nodes[prod_id]
        discount = data.get('discount', 0.0)
        discount_pct = f"{int(discount * 100)}%" if discount > 0 else "0%"
        enriched_recs.append({
            'product_id': prod_id,
            'probability': round(prob, 4),
            'color': data.get('color', 'N/A'),
            'size': data.get('size', 'N/A'),
            'category': data.get('category', 'N/A'),
            'stock': data.get('stock', 0),
            'discount': discount_pct,
            'message': f"🎯 You might like this {data.get('color', 'colorful')} {data.get('size', 'fit')} {data.get('category', 'item')} ({discount_pct} off!)"
        })
    return enriched_recs

# ———————————————————————
# 10. GET CUSTOMER DISPLAY NAME
# ———————————————————————
def get_display_name(G, cust_id):
    """Get readable customer name"""
    data = G.nodes[cust_id]
    if data.get('source') == 'international':
        return data.get('name', cust_id)
    elif data.get('source') == 'amazon':
        city = data.get('city', 'Unknown')
        country = data.get('country', 'Unknown')
        return f"{city}, {country}"
    else:
        return cust_id

# ———————————————————————
# 11. RUN SIMULATION
# ———————————————————————
sample_customers = [n for n, d in G.nodes(data=True) if d.get('label') == 'Customer'][:3]

simulation_dates = [
    datetime(2022, 1, 15),   # Low discount
    datetime(2022, 6, 20),   # Summer sale
    datetime(2022, 12, 15)   # Christmas
]

print("\n" + "="*80)
print("🎯 BALANCED ADAM-STYLE XGBOOST RECOMMENDATIONS")
print("   → 30% uniform distribution, 70% model bias")
print("   → Gentle temperature=2.0, smoothing=0.05")
print("   → Balanced category diversity (max 2 per category)")
print("="*80)

for customer_id in sample_customers:
    display_name = get_display_name(G, customer_id)
    print(f"\n🧑 Customer: {display_name}")
    print("─" * 80)

    for sim_date in simulation_dates:
        add_dynamic_discounts_with_tracking(G, sim_date)
        month_name = sim_date.strftime('%b')
        print(f"\n📅 {sim_date.strftime('%Y-%m-%d')} (Month: {month_name})")

        # Get recommendations with balanced Adam-style sampling
        recommendations = get_recommendations_for_customer(G, customer_id, sim_date, top_n=3)

        if not recommendations:
            print("  ⚠️ No recommendations available.")
            continue

        enriched_recs = enrich_recommendations(G, recommendations)
        for i, rec in enumerate(enriched_recs, 1):
            print(f"  {i}. {rec['color']} {rec['size']} {rec['category']} (P={rec['probability']}) {rec['discount']} off")
            print(f"     📦 Stock: {rec['stock']} | {rec['message']}")

print("\n" + "="*80)
print("🎉 BALANCED ADAM-STYLE XGBOOST RECOMMENDATIONS COMPLETE")
print("="*80)
