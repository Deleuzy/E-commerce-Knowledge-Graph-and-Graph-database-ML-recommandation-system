# === xgb_discount_softmax_enhanced.py ===
# Enhanced XGBoost recommendations with stronger softmax sampling
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import xgboost as xgb
import random
import os

print("🎯 Starting: ENHANCED XGBoost Discount + Softmax Sampling")

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
# 2. CONFIGURATION
# ———————————————————————
embedding_dim = 64
print(f"✅ Using embedding dimension: {embedding_dim}")

# ———————————————————————
# 3. TEMPORAL FEATURES
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
# 4. DYNAMIC DISCOUNT ASSIGNMENT (MORE EXTREME VARIATIONS)
# ———————————————————————
def add_dynamic_discounts(G, simulated_date):
    """Apply more extreme seasonal discounts for better variation"""
    for node, data in G.nodes(data=True):
        if data.get('label') == 'Product':
            month = simulated_date.month
            if month == 12:  # December: big sale
                discount = round(random.uniform(0.40, 0.60), 2)  # 40-60%
            elif month in [6, 7, 11]:  # Summer + Black Friday
                discount = round(random.uniform(0.20, 0.35), 2)  # 20-35%
            elif month in [1, 2, 8]:  # Clearance months
                discount = round(random.uniform(0.10, 0.25), 2)  # 10-25%
            else:  # Regular discount
                discount = round(random.uniform(0.0, 0.10), 2)  # 0-10%
            data['discount'] = discount

# ———————————————————————
# 5. FEATURE EXTRACTION (135 features)
# ———————————————————————
def get_link_features_with_discount(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time):
    """Extract features for XGBoost model with discount (135 features)"""
    try:
        c_row = cust_df[cust_df['node_id'] == cust_id].iloc[0]
        p_row = prod_df[prod_df['node_id'] == prod_id].iloc[0]
    except (IndexError, KeyError):
        return np.zeros(135)

    # Degrees
    customer_degree = float(c_row.get('degree', 0))
    product_degree = float(p_row.get('degree', 0))

    # Embeddings
    try:
        c_emb = [float(c_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
        p_emb = [float(p_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
    except:
        c_emb = [0.0] * emb_dim
        p_emb = [0.0] * emb_dim

    # Temporal features
    c_temp = calculate_temporal_features(G, cust_id, current_time)
    p_temp = calculate_temporal_features(G, prod_id, current_time)

    # Discount feature
    discount = float(G.nodes[prod_id].get('discount', 0.0))

    # Build feature vector - EXACTLY 135 features:
    features = np.array([
        customer_degree, product_degree,
        *c_emb, *p_emb,
        c_temp['recency'], c_temp['frequency'],
        p_temp['recency'], p_temp['frequency'],
        discount
    ])

    # Ensure exactly 135 features
    if len(features) != 135:
        if len(features) < 135:
            features = np.pad(features, (0, 135 - len(features)), constant_values=0.0)
        else:
            features = features[:135]

    return features

# ———————————————————————
# 6. GET VALID PRODUCTS
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
# 7. ENHANCED SOFTMAX SAMPLING (MORE DIVERSITY)
# ———————————————————————
def get_recommendations_for_customer(G, customer_id, current_time, top_n=3, temperature=4.0):
    """Enhanced recommendation engine with stronger softmax sampling"""
    if customer_id not in G:
        return []

    # Get products customer hasn't bought
    purchased_products = set()
    for _, order_id, data in G.edges(customer_id, data=True):
        if data.get('type') == 'PURCHASED' and G.nodes[order_id].get('label') == 'Order':
            for _, product_id, order_data in G.edges(order_id, data=True):
                if order_data.get('type') == 'CONTAINS' and G.nodes[product_id].get('label') == 'Product':
                    purchased_products.add(product_id)

    # Filter to products customer hasn't bought
    potential_products = [p for p in valid_products if p not in purchased_products]

    if len(potential_products) == 0:
        return []

    # Calculate features for a SAMPLE of potential products (for better diversity)
    # Take a random sample if too many products
    if len(potential_products) > 2000:
        potential_products = random.sample(potential_products, 2000)
        print(f"🔍 Sampling {len(potential_products)} products for diversity")

    product_features = []
    for product_id in potential_products:
        link_features = get_link_features_with_discount(
            customer_id, product_id,
            customer_features_df, product_features_df,
            embedding_dim, current_time
        )
        product_features.append((product_id, link_features))

    if not product_features:
        return []

    product_ids, feature_vectors = zip(*product_features)
    feature_vectors = np.array(feature_vectors)

    # Predict probabilities using XGBoost
    try:
        probs = xgb_wrapper.predict_proba(feature_vectors)[:, 1]
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return []

    # Enhanced softmax with higher temperature for more exploration
    # Clip probabilities to avoid log(0)
    clipped_probs = np.clip(probs, 1e-10, 1 - 1e-10)

    # Convert probabilities to logits
    logits = np.log(clipped_probs / (1 - clipped_probs))

    # Apply higher temperature for more diversity
    logits_scaled = logits / temperature
    # In your recommendation function, add this before softmax:
    # Add more noise to probabilities for diversity
    noise = np.random.normal(0, 0.1, len(probs))  # Add Gaussian noise
    noisy_probs = np.clip(probs + noise, 0.01, 0.99)  # Keep in reasonable range

# Then apply softmax to noisy_probs instead of original probs

    # Apply softmax
    exp_logits = np.exp(logits_scaled - np.max(logits_scaled))  # Numerical stability
    sampling_probs = exp_logits / exp_logits.sum()

    # Ensure no NaN values
    sampling_probs = np.nan_to_num(sampling_probs, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize again
    if sampling_probs.sum() > 0:
        sampling_probs = sampling_probs / sampling_probs.sum()
    else:
        # Fallback to uniform distribution
        sampling_probs = np.ones(len(product_ids)) / len(product_ids)

    # Sample with category diversity (more aggressive)
    seen_categories = set()
    selected = []
    attempts = 0
    max_attempts = top_n * 100  # More attempts for better sampling

    while len(selected) < top_n and attempts < max_attempts:
        if len(product_ids) == 0 or sampling_probs.sum() == 0:
            break

        # Sample based on softmax probabilities
        try:
            idx = np.random.choice(len(product_ids), p=sampling_probs)
        except ValueError:
            # Fallback if probabilities don't sum to 1
            idx = np.random.choice(len(product_ids))

        prod_id = product_ids[idx]
        prob = probs[idx]
        category = str(G.nodes[prod_id].get('category', 'Unknown')).upper().strip()

        # Check stock and category diversity
        if G.nodes[prod_id].get('stock', 0) <= 0:
            attempts += 1
            continue

        # More relaxed category diversity (allow some repetition if needed)
        if category in seen_categories and len(selected) < top_n:
            # Only skip if we have enough options
            if len(seen_categories) < min(5, len(product_ids) // 3):
                attempts += 1
                continue

        selected.append((prod_id, prob))
        seen_categories.add(category)
        attempts += 1

    # Fallback to top items if sampling didn't work well
    if len(selected) < top_n:
        print(f"⚠️ Only {len(selected)} items found via sampling. Using top recommendations...")
        # Sort by original probabilities
        prob_pairs = list(zip(product_ids, probs))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)

        for prod_id, prob in prob_pairs:
            if len(selected) >= top_n:
                break
            if G.nodes[prod_id].get('stock', 0) > 0:
                category = str(G.nodes[prod_id].get('category', 'Unknown')).upper().strip()
                if category not in seen_categories or len(seen_categories) < 3:
                    selected.append((prod_id, prob))
                    seen_categories.add(category)

    return selected

# ———————————————————————
# 8. ENRICH RECOMMENDATIONS
# ———————————————————————
def enrich_recommendations(G, recommendations):
    """Add human-readable information with discount awareness"""
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
# 9. GET CUSTOMER DISPLAY NAME
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
# 10. RUN SIMULATION WITH FIXED DATES
# ———————————————————————
sample_customers = [n for n, d in G.nodes(data=True) if d.get('label') == 'Customer'][:3]

# Use dates that should show clear discount differences
simulation_dates = [
    datetime(2022, 1, 15),   # Low discount period
    datetime(2022, 6, 20),   # Summer sale
    datetime(2022, 12, 15)   # High discount period (Christmas)
]

print("\n" + "="*80)
print("🎯 ENHANCED XGBOOST DISCOUNT + SOFTMAX RECOMMENDATIONS")
print("   → Higher temperature (3.0) = More exploration")
print("   → Extreme discounts = Clearer differences")
print("   → Better sampling = More varied results")
print("="*80)

for customer_id in sample_customers:
    display_name = get_display_name(G, customer_id)
    print(f"\n🧑 Customer: {display_name}")
    print("─" * 80)

    for sim_date in simulation_dates:
        # Add dynamic discounts based on date
        add_dynamic_discounts(G, sim_date)
        month_name = sim_date.strftime('%b')
        print(f"\n📅 {sim_date.strftime('%Y-%m-%d')} (Month: {month_name})")

        # Get recommendations with enhanced softmax sampling
        recommendations = get_recommendations_for_customer(G, customer_id, sim_date, top_n=3, temperature=3.0)

        if not recommendations:
            print("  ⚠️ No recommendations available.")
            continue

        # Enrich and display
        enriched_recs = enrich_recommendations(G, recommendations)
        for i, rec in enumerate(enriched_recs, 1):
            print(f"  {i}. {rec['color']} {rec['size']} {rec['category']} (P={rec['probability']}) {rec['discount']} off")
            print(f"     📦 Stock: {rec['stock']} | {rec['message']}")

print("\n" + "="*80)
print("🎉 ENHANCED XGBOOST DISCOUNT + SOFTMAX RECOMMENDATIONS COMPLETE")
print("="*80)
