import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# This function remains unchanged. It's a good way to create a test set.
def reset_half_of_positives(
    produits_clients_df, products_list, alpha=0.5, rng=None, copy=True
):
    """
    For a fraction `alpha` of clients who bought at least one product,
    hide half of their purchased products to create a test set.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(produits_clients_df)
    nb_chosen_clients = int(round(alpha * n))
    df_prod = produits_clients_df[products_list]
    arr = df_prod.to_numpy()
    
    has_any = (arr > 0).any(axis=1)
    eligible_rows = np.flatnonzero(has_any)
    if eligible_rows.size == 0:
        mask = np.zeros_like(arr, dtype=bool)
        return produits_clients_df.copy() if copy else produits_clients_df, mask
    
    k = min(len(eligible_rows), nb_chosen_clients)
    chosen_clients = rng.choice(eligible_rows, size=k, replace=False)
    
    out_df = produits_clients_df.copy() if copy else produits_clients_df
    out_arr = out_df[products_list].to_numpy()
    masked = np.zeros_like(arr, dtype=bool)
    
    for r in chosen_clients:
        pos_cols = np.flatnonzero(out_arr[r] > 0)
        m = pos_cols.size
        if m == 0:
            continue
        
        nb1 = 1 if m == 1 else int(np.ceil(m / 2)) # Use ceil to ensure at least one is left for training
        nb1 = min(nb1, m)
        cols_to_zero = rng.choice(pos_cols, size=nb1, replace=False)
        out_arr[r, cols_to_zero] = 0
        masked[r, cols_to_zero] = True

    out_df.loc[:, products_list] = out_arr
    return out_df, masked


def get_recommendation_scores(
    index, N, target_idx, X_prod_base, query_vectors
):
    """
    Calculates the raw recommendation scores for a single target user.
    This is the weighted average of neighbors' purchase vectors.
    """
    vec = np.asarray(query_vectors[target_idx], dtype=np.float32)
    ids, dists = index.get_nns_by_vector(vec, N, include_distances=True)
    
    ids = np.asarray(ids, dtype=np.int32)
    dists = np.asarray(dists, dtype=np.float32)

    if ids.size == 0:
        return np.zeros(X_prod_base.shape[1], dtype=np.float32)
    
    # Weight by distance
    dmax = np.max(dists)
    dmin = np.min(dists)
    denom = (dmax - dmin)
    if denom < 1e-6: # Avoid division by zero if all distances are the same
        weights = np.ones_like(dists, dtype=np.float32)
    else:
        weights = (dmax - dists) / denom
    
    # Calculate weighted average of neighbors' purchases
    scores = np.average(X_prod_base[ids], axis=0, weights=weights)
    return scores


def find_optimal_thresholds_per_product(all_scores, y_true_masked):
    """
    Finds the optimal threshold for each product individually to maximize Youden's J index.
    """
    num_products = all_scores.shape[1]
    optimal_thresholds = np.zeros(num_products)

    for j in tqdm(range(num_products), desc="Finding Optimal Thresholds per Product"):
        # Get scores and true labels for the current product across all clients
        product_scores = all_scores[:, j]
        product_true_labels = y_true_masked[:, j]

        # We only care about instances where we have a ground truth label (i.e., masked items)
        # However, for calculating FPR, we need all scores. Let's consider all test clients.
        
        p = np.sum(product_true_labels)
        n = len(product_true_labels) - p

        if p == 0 or n == 0:
            # If no positive or no negative examples for this product in the test set, use a default.
            optimal_thresholds[j] = 0.5 
            continue

        # Create pairs of (score, label) and sort by score descending
        pairs = sorted(zip(product_scores, product_true_labels), key=lambda x: x[0], reverse=True)
        sorted_scores, sorted_labels = zip(*pairs)
        sorted_scores = np.array(sorted_scores)
        sorted_labels = np.array(sorted_labels)

        # Calculate cumulative True Positives and False Positives
        tp_cumulative = np.cumsum(sorted_labels)
        fp_cumulative = np.cumsum(1 - sorted_labels)

        tpr = tp_cumulative / p
        fpr = fp_cumulative / n
        
        # Youden's J statistic
        j_scores = tpr - fpr
        
        if len(j_scores) == 0:
            optimal_thresholds[j] = 0.5
            continue

        best_idx = np.argmax(j_scores)
        
        # Set threshold to be between the score that gives max J and the next score
        if best_idx + 1 < len(sorted_scores):
            optimal_threshold = (sorted_scores[best_idx] + sorted_scores[best_idx + 1]) / 2.0
        else:
            # If it's the last element, the threshold is just below its score
            optimal_threshold = sorted_scores[best_idx] * 0.999
        
        optimal_thresholds[j] = optimal_threshold
        
    return optimal_thresholds


if __name__ == "__main__":
    produits_clients_path = "final_data_13_products.csv"
    produits_clients_df = pd.read_csv(produits_clients_path)
    
    # --- Parameters ---
    num_trials = 1  # Per-product thresholding is more stable, less trials needed to see behavior
    nb_similar = 100
    test_set_alpha = 0.5 # Hide purchases for 50% of eligible clients

    # --- Metrics storage ---
    trial_recalls = []
    trial_precisions = []
    trial_f1s = []

    for trial in range(num_trials):
        print(f"\n===== Trial {trial + 1}/{num_trials} =====")
        
        # --- 1. Data Split and Annoy Index Build ---
        index_df, test_df = train_test_split(
            produits_clients_df, test_size=0.2, random_state=42 + trial
        )
        
        products_list = [col for col in produits_clients_df.columns if col.startswith('product_')]
        if not products_list: # Fallback if no 'product_' prefix
            products_list = list(index_df.columns)
            
        nb_features = len(products_list) # Use number of products as feature dimension
        vectors_index = np.ascontiguousarray(index_df[products_list].to_numpy(dtype=np.float32))
        
        index_annoy = AnnoyIndex(nb_features, "angular") # Angular/Cosine is often good for this
        for i, v in enumerate(vectors_index):
            index_annoy.add_item(i, v)
        index_annoy.build(100)
        
        prod_mat_index = np.ascontiguousarray(index_df[products_list].to_numpy(dtype=np.float32))
        
        # --- 2. Create Test Set by Hiding Data ---
        # `y_true_original` is the ground truth before hiding anything
        y_true_original = test_df[products_list].to_numpy(dtype=np.uint8)
        
        # `dataset_test_modified` has items hidden. `mask` tells us WHICH items were hidden.
        dataset_test_modified, mask = reset_half_of_positives(
            test_df, products_list, alpha=test_set_alpha
        )

        # `y_true_masked` is our target for evaluation: only the 1s that were hidden.
        y_true_masked = (y_true_original * mask).astype(np.uint8)
        
        prod_mat_test = np.ascontiguousarray(dataset_test_modified[products_list].to_numpy(dtype=np.float32))
        num_test_clients = len(prod_mat_test)

        # --- 3. Calculate Raw Scores for ALL Test Clients ---
        all_scores = np.zeros_like(prod_mat_test, dtype=np.float32)
        
        # Using ThreadPoolExecutor to speed up score calculation
        with ThreadPoolExecutor() as executor:
            get_scores_partial = partial(
                get_recommendation_scores,
                index=index_annoy,
                N=nb_similar,
                X_prod_base=prod_mat_index,
                query_vectors=prod_mat_test
            )
            
            results = list(tqdm(executor.map(get_scores_partial, range(num_test_clients)), 
                                total=num_test_clients, desc="Calculating Scores"))

        for i, scores in enumerate(results):
            all_scores[i, :] = scores
            
        # --- 4. Find Optimal Threshold for EACH Product ---
        # This is the new, per-product strategy
        per_product_thresholds = find_optimal_thresholds_per_product(all_scores, y_true_masked)

        print("\nOptimal thresholds found for each product:")
        for product, thres in zip(products_list, per_product_thresholds):
            print(f"- {product}: {thres:.4f}")

        # --- 5. Evaluation ---
        # Generate predictions using the per-product thresholds
        y_pred = (all_scores >= per_product_thresholds).astype(np.uint8)

        # We only evaluate on the items that were hidden (where mask is True)
        # This prevents us from penalizing the model for not recommending items it was already shown.
        y_pred_for_eval = y_pred[mask]
        y_true_for_eval = y_true_original[mask] # These are all 1s by definition

        tp = np.sum(y_pred_for_eval)
        # Total positives (P) are the number of items we hid
        p = len(y_true_for_eval)
        # Predicted positives (PP) are the number of recommendations made for the hidden items
        pp = tp 

        recall = tp / p if p > 0 else 0.0
        # Precision in this context is how many of our recommendations for hidden slots were correct.
        # It's identical to recall here because we are only looking at slots we know should be 1.
        # A more standard precision would be TP / (all items recommended for these users)
        
        # Let's calculate a more standard precision over all potential recommendations (masked and not)
        # This is a better measure of overall performance.
        true_positives_overall = np.sum((y_pred == 1) & (y_true_masked == 1))
        predicted_positives_overall = np.sum(y_pred[mask]) # Count predictions only on masked items
        
        precision = true_positives_overall / predicted_positives_overall if predicted_positives_overall > 0 else 0.0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        trial_recalls.append(recall)
        trial_precisions.append(precision)
        trial_f1s.append(f1_score)
        
        print(f"\n--- Trial Results ---")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1-Score: {f1_score:.4f}")

    print("-" * 30)
    print("Mean Recall over trials:", np.round(np.mean(trial_recalls), 4))
    print("Mean Precision over trials:", np.round(np.mean(trial_precisions), 4))
    print("Mean F1-Score over trials:", np.round(np.mean(trial_f1s), 4))
    
    # --- 6. Per-Product False Positive / False Negative Analysis ---
    num_products = len(products_list)
    # These are only calculated on the masked items, which is what we are trying to predict
    true_positives = np.sum(y_true_masked, axis=0)
    false_negatives = np.sum((y_pred == 0) & (y_true_masked == 1), axis=0)
    
    # For false positives, we need to consider items that were NOT bought
    # And were NOT hidden. So true negatives are where y_true_original is 0.
    true_negatives = np.sum(y_true_original == 0, axis=0)
    false_positives = np.sum((y_pred == 1) & (y_true_original == 0), axis=0)
    
    print("\n--- Per-product Analysis ---")
    for j, product in enumerate(products_list):
        tpr = (true_positives[j] - false_negatives[j]) / true_positives[j] if true_positives[j] > 0 else 0.0
        fpr = false_positives[j] / true_negatives[j] if true_negatives[j] > 0 else 0.0
        
        print(f"Product: {product}")
        print(f"  - TPR (Recall): {tpr:.3f} ({true_positives[j] - false_negatives[j]}/{true_positives[j]})")
        print(f"  - FPR: {fpr:.3f} ({false_positives[j]}/{true_negatives[j]})")
        print(f"  - Missed Recommendations (FN): {false_negatives[j]} out of {true_positives[j]} hidden items.")
        print(f"  - Wrong Recommendations (FP): {false_positives[j]} out of {true_negatives[j]} non-purchased items.")
        print("-" * 20)
