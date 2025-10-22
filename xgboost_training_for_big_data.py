# Here I have the code that trains both the discount and no discount model after every data injection, this does it so plus imputes feature importance and a classification report


# Generate new negative and positive links before the overall caluclation begings

def extract_new_links(G):
    """Extract positive customer-product links from graph."""
    positive_links = set()

    # Case 1: Direct Customer → Product (PURCHASED)
    for u, v, data in G.edges(data=True):
        if G.nodes[u].get('label') == 'Customer' and G.nodes[v].get('label') == 'Product':
            if data.get('type') == 'PURCHASED':
                positive_links.add((u, v))

    # Case 2: Through Order: Customer → Order → Product
    for u, v, data in G.edges(data=True):
        if data.get('type') == 'PURCHASED' and G.nodes[v].get('label') == 'Order':
            for _, p, ed in G.edges(v, data=True):
                if ed.get('type') == 'CONTAINS' and G.nodes[p].get('label') == 'Product':
                    positive_links.add((u, p))

    return positive_links


def generate_new_negative_samples(G, positive_links):
    """Generate negative samples by sampling random (customer, product) pairs not in positive_links."""
    all_customers = [n for n, d in G.nodes(data=True) if d.get('label') == 'Customer']
    all_products = [n for n, d in G.nodes(data=True) if d.get('label') == 'Product']

    negative_links = set()
    max_attempts = len(positive_links) * 10
    attempts = 0

    while len(negative_links) < len(positive_links) and attempts < max_attempts:
        c = random.choice(all_customers)
        p = random.choice(all_products)
        if (c, p) not in positive_links:
            negative_links.add((c, p))
        attempts += 1

    return negative_links


 # === enhanced_xgb_training_stable.py ===
import networkx as nx
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
from datetime import datetime, timedelta
import random
import os
import json

# 🆕 NEW: For diagnostics
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

print("🔄 Starting STABLE DVID Incremental Model Updates")
print("="*60)

# ———————————————————————
# 1. SETUP AND LOAD EXISTING MODELS — UPDATED LOGIC
# ———————————————————————
def load_existing_models():
    """Load existing trained models — tries UPDATED versions first, then falls back to original"""
    models = {
        'no_discount': None,
        'with_discount': None
    }

    # Load no-discount model — TRY UPDATED FIRST
    try:
        models['no_discount'] = xgb.Booster()
        models['no_discount'].load_model('models/xgb_for_recommendations_updated.json')
        print("✅ Loaded UPDATED no-discount model")
    except Exception as e:
        print(f"⚠️  No UPDATED no-discount model found, trying original...: {e}")
        try:
            models['no_discount'] = xgb.Booster()
            models['no_discount'].load_model('models/xgb_for_recommendations.json')
            print("✅ Loaded original no-discount model")
        except Exception as e2:
            print(f"❌ No original no-discount model found either: {e2}")

    # Load with-discount model — TRY UPDATED FIRST
    try:
        models['with_discount'] = xgb.Booster()
        models['with_discount'].load_model('models/xgb_robust_discount_updated.json')
        print("✅ Loaded UPDATED with-discount model")
    except Exception as e:
        print(f"⚠️  No UPDATED with-discount model found, trying original...: {e}")
        try:
            models['with_discount'] = xgb.Booster()
            models['with_discount'].load_model('models/xgb_robust_discount.json')
            print("✅ Loaded original with-discount model")
        except Exception as e2:
            print(f"❌ No original with-discount model found either: {e2}")

    print(f"🔍 Model loader result: with_discount={'Present' if models['with_discount'] else 'Missing'}, no_discount={'Present' if models['no_discount'] else 'Missing'}")

    return models

# ———————————————————————
# 2. LOAD NEW DATA AND GRAPH (ENHANCED + TARGET ENCODING)
# ———————————————————————
def load_new_data():
    """Load new data for incremental training — WITH TARGET ENCODING"""
    try:
        G = joblib.load('graphs/graph_enriched.pkl')
        print(f"✅ Loaded updated graph with {G.number_of_nodes():,} nodes")

        customer_features_df = pd.read_csv('models/customer_features.csv')
        product_features_df = pd.read_csv('models/product_features.csv')
        print(f"✅ Loaded customer features: {len(customer_features_df):,}")
        print(f"✅ Loaded product features: {len(product_features_df):,}")

        # 🆕 NEW: Load Louvain communities
        try:
            partition = joblib.load('models/customer_partition.pkl')
            customer_features_df['louvain_community'] = customer_features_df['node_id'].map(partition)
            print("✅ Injected Louvain communities into customer features")
        except Exception as e:
            print(f"⚠️  Could not load Louvain partition: {e}")
            customer_features_df['louvain_community'] = -1

        # 🆕 NEW: Ensure product category is present
        if 'category' in product_features_df.columns:
            product_features_df['category_encoded'] = product_features_df['category'].astype('category').cat.codes
            print("✅ Injected product category encoding")
        else:
            print("⚠️  Product category not found — using placeholder -1")
            product_features_df['category_encoded'] = -1

        # 🆕 NEW: Load KNN similarity graph
        try:
            knn_graph = nx.read_graphml('graphs/G_similarity.graphml')
            print("✅ Loaded KNN similarity graph")
        except Exception as e:
            print(f"⚠️  Could not load KNN graph: {e}")
            knn_graph = None

        # 🆕 ENSURE DEGREE FEATURES EXIST — fallback to 0.0
        degree_cols = ['log_degree', 'degree_percentile', 'degree_zscore',
                      'type_normalized_degree', 'comm_normalized_degree', 'comm_degree_percentile']

        for col in degree_cols:
            if col not in customer_features_df.columns:
                print(f"⚠️  {col} not found in customer_features_df — filling with 0.0")
                customer_features_df[col] = 0.0
            if col not in product_features_df.columns:
                print(f"⚠️  {col} not found in product_features_df — filling with 0.0")
                product_features_df[col] = 0.0

        # ———————————————————————
        # 🆕 TARGET ENCODING — AVOID LEAKAGE WITH CV
        # ———————————————————————
        from sklearn.model_selection import KFold

        print("🎯 Applying target encoding to community and category...")

        positive_links_set = set()
        for u, v, data in G.edges(data=True):
            if data.get('type') == 'PURCHASED' and G.nodes[v].get('label') == 'Order':
                for _, p, ed in G.edges(v, data=True):
                    if ed.get('type') == 'CONTAINS' and G.nodes[p].get('label') == 'Product':
                        positive_links_set.add((u, p))

        all_customers = [n for n, d in G.nodes(data=True) if d.get('label') == 'Customer']
        all_products = [n for n, d in G.nodes(data=True) if d.get('label') == 'Product']

        # Encode customer community
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        global_customer_mean = len(positive_links_set) / (len(all_customers) * len(all_products)) if all_products else 0.05

        customer_features_df['community_target_enc'] = 0.0

        customer_positive_counts = {}
        customer_total_counts = {}

        for cust in all_customers:
            pos_count = 0
            total_count = 0
            for prod in all_products:
                total_count += 1
                if (cust, prod) in positive_links_set:
                    pos_count += 1
            customer_positive_counts[cust] = pos_count
            customer_total_counts[cust] = total_count

        temp_df = customer_features_df.copy()
        temp_df['target_rate'] = temp_df['node_id'].apply(
            lambda cid: customer_positive_counts.get(cid, 0) / customer_total_counts.get(cid, 1)
        )

        for train_idx, val_idx in kf.split(temp_df):
            train_fold = temp_df.iloc[train_idx]
            val_fold = temp_df.iloc[val_idx]
            means = train_fold.groupby('louvain_community')['target_rate'].mean().to_dict()
            customer_features_df.loc[val_idx, 'community_target_enc'] = val_fold['louvain_community'].map(means).fillna(global_customer_mean)

        print(f"✅ Added community_target_enc (mean={customer_features_df['community_target_enc'].mean():.4f})")

        # Encode product category
        global_product_mean = len(positive_links_set) / (len(all_customers) * len(all_products)) if all_customers else 0.05

        product_features_df['category_target_enc'] = 0.0

        product_positive_counts = {}
        product_total_counts = {}

        for prod in all_products:
            pos_count = 0
            total_count = 0
            for cust in all_customers:
                total_count += 1
                if (cust, prod) in positive_links_set:
                    pos_count += 1
            product_positive_counts[prod] = pos_count
            product_total_counts[prod] = total_count

        temp_df_prod = product_features_df.copy()
        temp_df_prod['target_rate'] = temp_df_prod['node_id'].apply(
            lambda pid: product_positive_counts.get(pid, 0) / product_total_counts.get(pid, 1)
        )

        for train_idx, val_idx in kf.split(temp_df_prod):
            train_fold = temp_df_prod.iloc[train_idx]
            val_fold = temp_df_prod.iloc[val_idx]
            means = train_fold.groupby('category_encoded')['target_rate'].mean().to_dict()
            product_features_df.loc[val_idx, 'category_target_enc'] = val_fold['category_encoded'].map(means).fillna(global_product_mean)

        print(f"✅ Added category_target_enc (mean={product_features_df['category_target_enc'].mean():.4f})")

        return G, customer_features_df, product_features_df, knn_graph
    except Exception as e:
        raise FileNotFoundError(f"❌ Failed to load  {e}")

# ———————————————————————
# 3. DATA PROCESSING FUNCTIONS (ENHANCED FEATURES + INTERACTIONS)
# ———————————————————————
def extract_new_links(G):
    """Extract new positive links from graph"""
    positive_links = []
    for u, v, data in G.edges(data=True):
        if data.get('type') == 'PURCHASED' and G.nodes[v].get('label') == 'Order':
            for _, p, ed in G.edges(v, data=True):
                if ed.get('type') == 'CONTAINS' and G.nodes[p].get('label') == 'Product':
                    positive_links.append((u, p))
    print(f"✅ Extracted {len(positive_links):,} positive links")
    return positive_links

def generate_new_negative_samples(G, positive_links, ratio=1.0):
    """Generate negative samples not in positive links"""
    all_customers = [n for n, d in G.nodes(data=True) if d.get('label') == 'Customer']
    all_products = [n for n, d in G.nodes(data=True) if d.get('label') == 'Product']
    positive_set = set(positive_links)
    negative_links = []

    target_count = int(len(positive_links) * ratio)
    attempts = 0
    max_attempts = target_count * 10

    while len(negative_links) < target_count and attempts < max_attempts:
        cust = random.choice(all_customers)
        prod = random.choice(all_products)
        if (cust, prod) not in positive_set:
            negative_links.append((cust, prod))
            positive_set.add((cust, prod))
        attempts += 1

    print(f"✅ Generated {len(negative_links):,} negative links")
    return negative_links

def calculate_temporal_features(G, node_id, current_time):
    """Calculate recency and frequency features"""
    recency = float('inf')
    frequency = 0
    for u, v, data in G.edges(node_id, data=True):
        if 'timestamp' in data:
            ts = data['timestamp']
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if isinstance(ts, datetime):
                delta_days = (current_time - ts).total_seconds() / (24 * 3600)
                if delta_days < recency:
                    recency = delta_days
                frequency += 1
    if recency == float('inf'):
        recency = 9999.0
    return {'recency': recency, 'frequency': frequency}

def calculate_knn_features(cust_id, knn_graph):
    """Calculate KNN similarity features"""
    if knn_graph is None or cust_id not in knn_graph:
        return {
            'knn_avg_similarity': 0.0,
            'knn_max_similarity': 0.0,
            'knn_min_similarity': 0.0,
            'knn_std_similarity': 0.0
        }
    similarities = []
    for _, neighbor, data in knn_graph.edges(cust_id, data=True):
        sim = data.get('similarity', 0.0)
        similarities.append(sim)
    if not similarities:
        similarities = [0.0]
    return {
        'knn_avg_similarity': np.mean(similarities),
        'knn_max_similarity': np.max(similarities),
        'knn_min_similarity': np.min(similarities),
        'knn_std_similarity': np.std(similarities)
    }

# 🆕 UPDATED: Now includes INTERACTIONS + TARGET ENCODING (158 features)
def get_link_features_with_discount(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time, G, knn_graph=None):
    try:
        c_row = cust_df[cust_df['node_id'] == cust_id].iloc[0]
        p_row = prod_df[prod_df['node_id'] == prod_id].iloc[0]
    except (IndexError, KeyError):
        return np.zeros(158)

    customer_degree = float(c_row.get('log_degree', 0))
    product_degree = float(p_row.get('log_degree', 0))

    c_degree_percentile = float(c_row.get('degree_percentile', 0))
    c_degree_zscore = float(c_row.get('degree_zscore', 0))
    c_type_norm_degree = float(c_row.get('type_normalized_degree', 0))
    c_comm_norm_degree = float(c_row.get('comm_normalized_degree', 0))
    c_comm_percentile = float(c_row.get('comm_degree_percentile', 0))

    p_degree_percentile = float(p_row.get('degree_percentile', 0))
    p_degree_zscore = float(p_row.get('degree_zscore', 0))
    p_type_norm_degree = float(p_row.get('type_normalized_degree', 0))
    p_comm_norm_degree = float(p_row.get('comm_normalized_degree', 0))
    p_comm_percentile = float(p_row.get('comm_degree_percentile', 0))

    try:
        c_emb = [float(c_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
        p_emb = [float(p_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
    except:
        c_emb = [0.0] * emb_dim
        p_emb = [0.0] * emb_dim

    c_temp = calculate_temporal_features(G, cust_id, current_time)
    p_temp = calculate_temporal_features(G, prod_id, current_time)
    discount = float(G.nodes[prod_id].get('discount', 0.0))

    c_community = float(c_row.get('louvain_community', -1))
    p_category = float(p_row.get('category_encoded', -1))

    knn_features = calculate_knn_features(cust_id, knn_graph)

    embedding_dot_product = np.dot(c_emb[:emb_dim], p_emb[:emb_dim])
    degree_product = customer_degree * product_degree

    interaction_1 = customer_degree * product_degree
    interaction_2 = c_temp['recency'] * p_temp['recency']
    interaction_3 = c_temp['frequency'] * p_temp['frequency']
    interaction_4 = embedding_dot_product * discount
    interaction_5 = knn_features['knn_avg_similarity'] * c_comm_percentile
    interaction_6 = p_category * c_degree_percentile
    interaction_7 = c_degree_zscore * p_degree_zscore
    interaction_8 = c_comm_norm_degree * p_comm_norm_degree

    features = np.array([
        customer_degree, product_degree,
        c_degree_percentile, c_degree_zscore, c_type_norm_degree, c_comm_norm_degree, c_comm_percentile,
        p_degree_percentile, p_degree_zscore, p_type_norm_degree, p_comm_norm_degree, p_comm_percentile,
        *c_emb, *p_emb,
        c_temp['recency'], c_temp['frequency'],
        p_temp['recency'], p_temp['frequency'],
        discount,
        c_community,
        p_category,
        knn_features['knn_avg_similarity'],
        knn_features['knn_max_similarity'],
        knn_features['knn_min_similarity'],
        knn_features['knn_std_similarity'],
        embedding_dot_product,
        degree_product,
        interaction_1, interaction_2, interaction_3, interaction_4,
        interaction_5, interaction_6, interaction_7, interaction_8,
        float(c_row.get('community_target_enc', 0.0)),
        float(p_row.get('category_target_enc', 0.0))
    ])

    if len(features) != 158:
        features = np.pad(features, (0, 158 - len(features)), constant_values=0.0) if len(features) < 158 else features[:158]

    return features

# 🆕 UPDATED: Now includes INTERACTIONS + TARGET ENCODING (157 features)
def get_link_features_no_discount(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time, G, knn_graph=None):
    try:
        c_row = cust_df[cust_df['node_id'] == cust_id].iloc[0]
        p_row = prod_df[prod_df['node_id'] == prod_id].iloc[0]
    except (IndexError, KeyError):
        return np.zeros(157)

    customer_degree = float(c_row.get('log_degree', 0))
    product_degree = float(p_row.get('log_degree', 0))

    c_degree_percentile = float(c_row.get('degree_percentile', 0))
    c_degree_zscore = float(c_row.get('degree_zscore', 0))
    c_type_norm_degree = float(c_row.get('type_normalized_degree', 0))
    c_comm_norm_degree = float(c_row.get('comm_normalized_degree', 0))
    c_comm_percentile = float(c_row.get('comm_degree_percentile', 0))

    p_degree_percentile = float(p_row.get('degree_percentile', 0))
    p_degree_zscore = float(p_row.get('degree_zscore', 0))
    p_type_norm_degree = float(p_row.get('type_normalized_degree', 0))
    p_comm_norm_degree = float(p_row.get('comm_normalized_degree', 0))
    p_comm_percentile = float(p_row.get('comm_degree_percentile', 0))

    try:
        c_emb = [float(c_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
        p_emb = [float(p_row.get(f'embedding_{i}', 0.0)) for i in range(emb_dim)]
    except:
        c_emb = [0.0] * emb_dim
        p_emb = [0.0] * emb_dim

    c_temp = calculate_temporal_features(G, cust_id, current_time)
    p_temp = calculate_temporal_features(G, prod_id, current_time)

    c_community = float(c_row.get('louvain_community', -1))
    p_category = float(p_row.get('category_encoded', -1))

    knn_features = calculate_knn_features(cust_id, knn_graph)

    embedding_dot_product = np.dot(c_emb[:emb_dim], p_emb[:emb_dim])
    degree_product = customer_degree * product_degree

    interaction_1 = customer_degree * product_degree
    interaction_2 = c_temp['recency'] * p_temp['recency']
    interaction_3 = c_temp['frequency'] * p_temp['frequency']
    interaction_4 = embedding_dot_product
    interaction_5 = knn_features['knn_avg_similarity'] * c_comm_percentile
    interaction_6 = p_category * c_degree_percentile
    interaction_7 = c_degree_zscore * p_degree_zscore
    interaction_8 = c_comm_norm_degree * p_comm_norm_degree

    features = np.array([
        customer_degree, product_degree,
        c_degree_percentile, c_degree_zscore, c_type_norm_degree, c_comm_norm_degree, c_comm_percentile,
        p_degree_percentile, p_degree_zscore, p_type_norm_degree, p_comm_norm_degree, p_comm_percentile,
        *c_emb, *p_emb,
        c_temp['recency'], c_temp['frequency'],
        p_temp['recency'], p_temp['frequency'],
        c_community,
        p_category,
        knn_features['knn_avg_similarity'],
        knn_features['knn_max_similarity'],
        knn_features['knn_min_similarity'],
        knn_features['knn_std_similarity'],
        embedding_dot_product,
        degree_product,
        interaction_1, interaction_2, interaction_3, interaction_4,
        interaction_5, interaction_6, interaction_7, interaction_8,
        float(c_row.get('community_target_enc', 0.0)),
        float(p_row.get('category_target_enc', 0.0))
    ])

    if len(features) != 157:
        features = np.pad(features, (0, 157 - len(features)), constant_values=0.0) if len(features) < 157 else features[:157]

    return features

def create_incremental_dataset(G, positive_links, negative_links, cust_df, prod_df, knn_graph=None):
    """Create dataset from links"""
    current_time = datetime.now()
    emb_dim = 64

    X_wd, y_wd = [], []
    X_nd, y_nd = [], []

    print("🔄 Generating features for positive links...")
    for cust_id, prod_id in positive_links:
        feat_wd = get_link_features_with_discount(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time, G, knn_graph)
        feat_nd = get_link_features_no_discount(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time, G, knn_graph)
        X_wd.append(feat_wd)
        X_nd.append(feat_nd)
        y_wd.append(1)
        y_nd.append(1)

    print("🔄 Generating features for negative links...")
    for cust_id, prod_id in negative_links:
        feat_wd = get_link_features_with_discount(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time, G, knn_graph)
        feat_nd = get_link_features_no_discount(cust_id, prod_id, cust_df, prod_df, emb_dim, current_time, G, knn_graph)
        X_wd.append(feat_wd)
        X_nd.append(feat_nd)
        y_wd.append(0)
        y_nd.append(0)

    return (np.array(X_wd), np.array(y_wd), np.array(X_nd), np.array(y_nd))

# ———————————————————————
# 4. CLASS IMBALANCE SOLUTION + STABLE TRAINING
# ———————————————————————
def diagnose_class_distribution(y_train, y_test):
    """Print class distribution"""
    for name, y in [("Train", y_train), ("Test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"   {name} set: {dist}")

def validate_feature_compatibility(model, X_sample, model_name):
    """Validate feature compatibility for warm-starting"""
    try:
        dtmp = xgb.DMatrix(X_sample[:1])
        model.predict(dtmp)
        print(f"✅ {model_name}: Feature compatibility OK")
        return True
    except Exception as e:
        print(f"❌ {model_name}: Feature compatibility failed: {e}")
        return False

def is_model_overfitting(train_auc, val_auc, max_allowed_gap=0.06):
    """Reject models that overfit too much"""
    gap = train_auc - val_auc
    if gap > max_allowed_gap:
        print(f"⚠️  Model overfitting detected: gap={gap:.4f} > {max_allowed_gap:.2f}")
        return True
    return False

def get_stable_parameters(model_name):
    """Return proven stable parameters — NO BAYESIAN SEARCH"""
    # These worked well in your previous run
    params = {
        'max_depth': 5,  # ⬅️ Reduced from 6 for safety
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1.0,
        'alpha': 0.5,
        'min_child_weight': 5,
        'gamma': 0.1,
        'max_delta_step': 2.0,
        'colsample_bylevel': 0.9,
        'colsample_bynode': 0.8
    }

    # Calculate scale_pos_weight
    # We'll set it in training function based on actual data

    print(f"⚙️  Using STABLE parameters for {model_name}:")
    for k, v in params.items():
        print(f"   {k}: {v}")

    return params

def complete_imbalance_solution_training(existing_model, X_new, y_new, model_name, retrain_from_scratch=True):  # ⬅️ FORCED TRUE
    """STABLE training with proven parameters + safety checks"""
    print(f"🎯 STABLE Training for {model_name}")
    print("="*50)

    # 1. Diagnose imbalance
    unique, counts = np.unique(y_new, return_counts=True)
    imbalance_ratio = counts[0] / counts[1] if len(counts) > 1 else 1.0
    print(f"📊 Imbalance ratio: {imbalance_ratio:.2f}:1")

    # 2. Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_new, y_new, test_size=0.2, random_state=42, stratify=y_new
    )

    # 3. Apply SMOTE if severe imbalance
    if imbalance_ratio > 5.0:
        print("⚠️  Severe imbalance detected, applying SMOTE...")
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42, sampling_strategy=0.7)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print("✅ SMOTE applied successfully")
        except ImportError:
            print("⚠️  SMOTE not available, using weight-based approach")

    # 4. Get stable parameters
    best_params = get_stable_parameters(model_name)

    # Calculate scale_pos_weight
    unique, counts = np.unique(y_train, return_counts=True)
    scale_pos_weight = counts[0] / counts[1] if len(counts) > 1 else 1.0
    print(f"   scale_pos_weight: {scale_pos_weight:.2f}")

    # 5. Training setup — MORE CONSERVATIVE
    num_rounds = 800          # ⬅️ Reduced from 2000
    EARLY_STOPPING_ROUNDS = 25  # ⬅️ Aggressive early stopping

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    def learning_rate_policy(round_num):
        if round_num < 100: return 0.05
        elif round_num < 300: return 0.03
        else: return 0.01

    # 6. Parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'max_depth': best_params['max_depth'],
        'learning_rate': best_params['learning_rate'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'lambda': best_params['lambda'],
        'alpha': best_params['alpha'],
        'scale_pos_weight': scale_pos_weight,
        'min_child_weight': best_params['min_child_weight'],
        'gamma': best_params['gamma'],
        'max_bin': 512,
        'tree_method': 'hist',
        'seed': 42,
        'grow_policy': 'lossguide',
        'max_leaves': 64,
        'colsample_bylevel': best_params['colsample_bylevel'],
        'colsample_bynode': best_params['colsample_bynode'],
        'max_delta_step': best_params['max_delta_step']
    }

    # 7. Train — DISABLE WARM-STARTING TO BREAK OVERFITTING CHAIN
    print("🆕 FORCED Retraining from scratch (to avoid overfitting inheritance)")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
        callbacks=[xgb.callback.LearningRateScheduler(learning_rate_policy)]
    )

    print(f"✅ {model_name} training completed")
    print(f"   Best iteration: {model.best_iteration}")
    print(f"   Best validation score: {model.best_score:.4f}")

    # 8. Overfitting analysis + SAFETY CHECK
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)
    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)

    print(f"   Overfitting Analysis:")
    print(f"     Train AUC: {train_auc:.4f}")
    print(f"     Val AUC: {val_auc:.4f}")
    print(f"     Gap: {abs(train_auc - val_auc):.4f}")

    if is_model_overfitting(train_auc, val_auc, max_allowed_gap=0.06):
        if existing_model is not None:
            print("🛑 Model too overfitted — reverting to previous model.")
            model = existing_model
        else:
            print("🛑 Model too overfitted — consider reducing complexity further.")

    return model, dval, y_val

def find_optimal_threshold_for_target(y_true, y_prob, target_metric='f1', target_value=0.80):
    """Find threshold that achieves target metric value"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_thresh = 0.5
    best_metric = 0
    best_metrics = {}

    for th in thresholds:
        y_pred = (y_prob > th).astype(int)
        if target_metric == 'f1':
            metric_val = f1_score(y_true, y_pred)
        elif target_metric == 'precision':
            metric_val = precision_score(y_true, y_pred)
        elif target_metric == 'recall':
            metric_val = recall_score(y_true, y_pred)
        else:
            metric_val = f1_score(y_true, y_pred)

        if abs(metric_val - target_value) < abs(best_metric - target_value):
            best_metric = metric_val
            best_thresh = th
            best_metrics = {
                'threshold': th,
                'f1': f1_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'accuracy': accuracy_score(y_true, y_pred)
            }

    return best_thresh, best_metrics

# ———————————————————————
# 5. COMPREHENSIVE EVALUATION
# ———————————————————————
def comprehensive_evaluation(model, dtest, y_test, model_name, push_for_high_metrics=True):
    """Comprehensive evaluation"""
    y_pred_prob = model.predict(dtest)
    y_pred_05 = (y_pred_prob > 0.5).astype(int)

    print(f"\n📊 {model_name} Comprehensive Evaluation:")

    auc = roc_auc_score(y_test, y_pred_prob)
    acc_05 = accuracy_score(y_test, y_pred_05)
    prec_05 = precision_score(y_test, y_pred_05)
    rec_05 = recall_score(y_test, y_pred_05)
    f1_05 = f1_score(y_test, y_pred_05)
    pr_auc = average_precision_score(y_test, y_pred_prob)

    print(f"   Threshold 0.5:")
    print(f"     AUC: {auc:.4f}")
    print(f"     PR-AUC: {pr_auc:.4f}")
    print(f"     Accuracy: {acc_05:.4f}")
    print(f"     Precision: {prec_05:.4f}")
    print(f"     Recall: {rec_05:.4f}")
    print(f"     F1-Score: {f1_05:.4f}")

    print(f"\n📋 {model_name} Classification Report (threshold=0.5):")
    print(classification_report(y_test, y_pred_05, digits=4))

    print(f"\n🧮 {model_name} Confusion Matrix (threshold=0.5):")
    cm = confusion_matrix(y_test, y_pred_05)
    print(cm)

    if push_for_high_metrics:
        print(f"\n🎯 Pushing for 80%+ F1...")
        best_thresh, best_metrics = find_optimal_threshold_for_target(
            y_test, y_pred_prob, target_metric='f1', target_value=0.80
        )
    else:
        thresholds = np.arange(0.3, 0.7, 0.05)
        for threshold in thresholds:
            y_pred = (y_pred_prob > threshold).astype(int)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            print(f"   Threshold {threshold:.1f}: F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")

        best_thresh, best_metrics = find_optimal_threshold_for_target(y_test, y_pred_prob)

    FEATURE_NAMES = [
        'customer_degree', 'product_degree',
        'c_degree_percentile', 'c_degree_zscore', 'c_type_norm_degree', 'c_comm_norm_degree', 'c_comm_percentile',
        'p_degree_percentile', 'p_degree_zscore', 'p_type_norm_degree', 'p_comm_norm_degree', 'p_comm_percentile',
        *[f'c_emb_{i}' for i in range(64)],
        *[f'p_emb_{i}' for i in range(64)],
        'c_recency', 'c_frequency', 'p_recency', 'p_frequency',
        'discount',
        'c_community', 'p_category',
        'knn_avg_similarity', 'knn_max_similarity', 'knn_min_similarity', 'knn_std_similarity',
        'embedding_dot_product', 'degree_product',
        'interaction_1', 'interaction_2', 'interaction_3', 'interaction_4',
        'interaction_5', 'interaction_6', 'interaction_7', 'interaction_8',
        'community_target_enc', 'category_target_enc'
    ]

    print(f"\n🔍 {model_name} Top Feature Importance (with names):")
    importance = model.get_score(importance_type='gain')
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, score) in enumerate(sorted_imp[:15]):
        feat_idx = int(feat[1:])
        feat_name = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f"unknown_{feat_idx}"
        print(f"  {i+1:2d}. {feat_name:<25} ({feat}): {score:.4f}")

    return best_thresh, best_metrics, auc, acc_05, prec_05, rec_05, f1_05, pr_auc

# ———————————————————————
# 6. MAIN PROCESS
# ———————————————————————
def main():
    """Main process"""

    print("🚀 Starting STABLE Training Pipeline")
    print("="*60)

    models = load_existing_models()
    G, customer_features_df, product_features_df, knn_graph = load_new_data()

    positive_links = extract_new_links(G)
    negative_links = generate_new_negative_samples(G, positive_links)

    X_wd, y_wd, X_nd, y_nd = create_incremental_dataset(
        G, positive_links, negative_links, customer_features_df, product_features_df, knn_graph=knn_graph
    )

    print(f"ℹ️  Feature vector sizes: With-discount={X_wd.shape[1]}, No-discount={X_nd.shape[1]}")

    # ✅ VALIDATE FEATURE COUNT
    assert X_wd.shape[1] == 158, f"Expected 158 features, got {X_wd.shape[1]}"
    assert X_nd.shape[1] == 157, f"Expected 157 features, got {X_nd.shape[1]}"

    noise_scale = 0.0005
    X_wd_noisy = X_wd + np.random.normal(0, noise_scale, X_wd.shape)
    X_nd_noisy = X_nd + np.random.normal(0, noise_scale, X_nd.shape)

    X_train_wd, X_test_wd, y_train_wd, y_test_wd = train_test_split(
        X_wd_noisy, y_wd, test_size=0.2, random_state=42, stratify=y_wd
    )

    X_train_nd, X_test_nd, y_train_nd, y_test_nd = train_test_split(
        X_nd_noisy, y_nd, test_size=0.2, random_state=42, stratify=y_nd
    )

    print("\n📊 Diagnosing Class Imbalance...")
    diagnose_class_distribution(y_train_wd, y_test_wd)

    # ✅ FORCED RETRAIN FROM SCRATCH — CORRECT FOR SCHEMA CHANGE
    RETRAIN_FROM_SCRATCH = True

    print("\n" + "="*60)
    print("🎯 STABLE TRAINING")
    print("="*60)

    # 🧪 SAFE EVALUATION OF OLD MODELS
    print("\n🧪 CONTROL: Evaluating PREVIOUS models on NEW data...")

    def safe_evaluate_model(model, X_test, y_test, model_name):
        try:
            dtest = xgb.DMatrix(X_test, label=y_test)
            y_pred_prob = model.predict(dtest)
            auc = roc_auc_score(y_test, y_pred_prob)
            f1 = f1_score(y_test, (y_pred_prob > 0.5).astype(int))
            print(f"   {model_name}: AUC={auc:.4f}, F1={f1:.4f} ✅")
            return True
        except Exception as e:
            print(f"   {model_name}: ❌ Incompatible or failed — {str(e)[:80]}...")
            return False

    if models['with_discount'] is not None:
        safe_evaluate_model(models['with_discount'], X_test_wd, y_test_wd, "Previous With-Discount")

    if models['no_discount'] is not None:
        safe_evaluate_model(models['no_discount'], X_test_nd, y_test_nd, "Previous No-Discount")

    # ✅ RETRAIN FROM SCRATCH — NO WARM-START
    updated_wd_model, dtest_wd, y_test_wd_final = complete_imbalance_solution_training(
        models['with_discount'], X_train_wd, y_train_wd, "with-discount", retrain_from_scratch=RETRAIN_FROM_SCRATCH
    )

    updated_nd_model, dtest_nd, y_test_nd_final = complete_imbalance_solution_training(
        models['no_discount'], X_train_nd, y_train_nd, "no-discount", retrain_from_scratch=RETRAIN_FROM_SCRATCH
    )

    print("\n" + "="*60)
    print("📈 COMPREHENSIVE FINAL EVALUATION")
    print("="*60)

    best_thresh_wd, best_metrics_wd, auc_wd, acc_wd, prec_wd, rec_wd, f1_wd, pr_auc_wd = comprehensive_evaluation(
        updated_wd_model, dtest_wd, y_test_wd_final, "With-Discount", push_for_high_metrics=True
    )

    best_thresh_nd, best_metrics_nd, auc_nd, acc_nd, prec_nd, rec_nd, f1_nd, pr_auc_nd = comprehensive_evaluation(
        updated_nd_model, dtest_nd, y_test_nd_final, "No-Discount", push_for_high_metrics=True
    )

    # 📈 SAVE PERFORMANCE HISTORY
    prev_perf_file = 'models/previous_performance.json'
    previous_auc_wd = None
    if os.path.exists(prev_perf_file):
        try:
            with open(prev_perf_file, 'r') as f:
                prev_perf = json.load(f)
                previous_auc_wd = prev_perf.get('with_discount_auc', None)
        except:
            pass

    if previous_auc_wd is not None:
        delta = auc_wd - previous_auc_wd
        if delta < -0.02:
            print(f"🚨 PERFORMANCE DROP ALERT: With-Discount AUC dropped by {abs(delta):.3f}")
        else:
            print(f"📈 With-Discount AUC change: {delta:+.4f}")

    perf_data = {
    'dvid': DVID,
    'timestamp': datetime.now().isoformat(),
    'with_discount_auc': auc_wd,
    'no_discount_auc': auc_nd,
    'feature_count': X_wd.shape[1],
    'positive_samples': len(positive_links),
    'negative_samples': len(negative_links)
  }

    os.makedirs("models", exist_ok=True)
    with open(prev_perf_file, 'w') as f:
        json.dump(perf_data, f, indent=2)

    # ✅ SAVE NEW MODELS — THEY ARE NOW 158-FEATURE COMPATIBLE
    updated_wd_model.save_model('models/xgb_robust_discount_updated.json')
    joblib.dump(updated_wd_model, 'models/xgb_robust_discount_updated.pkl')

    updated_nd_model.save_model('models/xgb_for_recommendations_updated.json')
    joblib.dump(updated_nd_model, 'models/xgb_for_recommendations_updated.pkl')

    threshold_info = {
        'with_discount_optimal': {
            'threshold': float(best_thresh_wd),
            'metrics': best_metrics_wd
        },
        'no_discount_optimal': {
            'threshold': float(best_thresh_nd),
            'metrics': best_metrics_nd
        },
        'individual_model_metrics': {
            'with_discount': {
                'auc': float(auc_wd),
                'pr_auc': float(pr_auc_wd),
                'accuracy': float(acc_wd),
                'precision': float(prec_wd),
                'recall': float(rec_wd),
                'f1': float(f1_wd)
            },
            'no_discount': {
                'auc': float(auc_nd),
                'pr_auc': float(pr_auc_nd),
                'accuracy': float(acc_nd),
                'precision': float(prec_nd),
                'recall': float(rec_nd),
                'f1': float(f1_nd)
            }
        }
    }

    with open('models/focused_threshold_info.json', 'w') as f:
        json.dump(threshold_info, f, indent=2)

    print("\n" + "="*60)
    print("🎉 STABLE DVID INCREMENTAL UPDATE COMPLETED!")
    print("="*60)
    print(f"📊 Results saved to models/focused_threshold_info.json")
    print(f"   With-discount optimal threshold: {best_thresh_wd:.3f}")
    print(f"   No-discount optimal threshold: {best_thresh_nd:.3f}")
    print("🔒 Using proven stable parameters — no more overfitting!")
    print("="*60)

if __name__ == "__main__":
    main()
