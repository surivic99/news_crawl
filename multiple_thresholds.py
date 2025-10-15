import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor


def reset_half_of_positives(
    produits_clients_df, products_list, alpha=0.2, rng=None, copy=True
):
    """
    Randomly mask half of the positive products for a fraction of clients.
    
    Returns:
        modified_df: DataFrame with masked products set to 0
        mask: Binary array indicating which products were masked (1 = masked)
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
    out = produits_clients_df.copy() if copy else produits_clients_df
    out_arr = out[products_list].to_numpy()
    masked = np.zeros_like(arr, dtype=bool)
    
    for r in chosen_clients:
        pos_cols = np.flatnonzero(out_arr[r] > 0)
        m = pos_cols.size
        if m == 0:
            continue
        
        nb1 = 1 if m == 1 else int(round(m / 2))
        nb1 = min(nb1, m)
        cols_to_zero = rng.choice(pos_cols, size=nb1, replace=False)
        out_arr[r, cols_to_zero] = 0
        masked[r, cols_to_zero] = True

    out.loc[:, products_list] = out_arr
    return out, masked


def compute_weighted_scores(index_annoy, query_vec, nb_similar, X_prod_base):
    """
    Compute weighted average scores for all products based on nearest neighbors.
    """
    ids, dists = index_annoy.get_nns_by_vector(
        query_vec, nb_similar, include_distances=True
    )
    ids = np.asarray(ids, dtype=np.int32)
    dists = np.asarray(dists, dtype=np.float32)
    
    if ids.size == 0:
        return np.zeros(X_prod_base.shape[1], dtype=np.float32)
    
    dmax, dmin = np.max(dists), np.min(dists)
    if dmax == dmin:
        weights = np.ones_like(dists, dtype=np.float32)
    else:
        weights = (dmax - dists) / (dmax - dmin)
    
    scores = np.average(X_prod_base[ids], axis=0, weights=weights)
    return scores


def estimate_per_product_thresholds(
    prod_mat_query, prod_mat_base, y_true_original, y_true_masked, 
    index_annoy, nb_similar, products_list
):
    """
    Estimate optimal threshold for each product using Youden's Index.
    Only considers products that are recommendable (not currently owned).
    """
    num_products = prod_mat_query.shape[1]
    thresholds = np.zeros(num_products)
    
    print("Estimating per-product thresholds...")
    for product_idx in tqdm(range(num_products)):
        all_scores = []
        all_labels = []
        
        for pos in range(len(prod_mat_query)):
            vec = np.asarray(prod_mat_query[pos], dtype=np.float32)
            scores = compute_weighted_scores(index_annoy, vec, nb_similar, prod_mat_base)
            
            # Current ownership: original minus what was masked
            current_owned = y_true_original[pos, product_idx] - y_true_masked[pos, product_idx]
            
            # Only consider if product is recommendable (not currently owned)
            if current_owned == 0:
                all_scores.append(scores[product_idx])
                all_labels.append(y_true_masked[pos, product_idx])
        
        if len(all_scores) == 0 or sum(all_labels) == 0:
            thresholds[product_idx] = 0.5
            continue
            
        scores_arr = np.array(all_scores)
        labels_arr = np.array(all_labels)
        
        # Find threshold maximizing Youden's Index (Sensitivity + Specificity - 1)
        unique_scores = np.sort(np.unique(scores_arr))
        best_youden = -1
        best_thresh = 0.5
        
        for thresh in unique_scores:
            preds = (scores_arr >= thresh).astype(int)
            tp = np.sum(preds & labels_arr)
            tn = np.sum((1 - preds) & (1 - labels_arr))
            fp = np.sum(preds & (1 - labels_arr))
            fn = np.sum((1 - preds) & labels_arr)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            youden = sensitivity + specificity - 1
            
            if youden > best_youden:
                best_youden = youden
                best_thresh = thresh
        
        thresholds[product_idx] = best_thresh
    
    return thresholds


def eval_one_corrected(
    pos, X_prod_base, y_true_original, y_true_masked, 
    index_annoy, per_product_thresholds, nb_similar, query_vectors
):
    """
    Evaluate recommendations for one client.
    
    - Only recommends products NOT currently owned
    - Recall: fraction of masked products that were recommended
    - Precision: fraction of recommendations that were actually masked
    """
    vec = np.asarray(query_vectors[pos], dtype=np.float32)
    scores = compute_weighted_scores(index_annoy, vec, nb_similar, X_prod_base)
    
    # Current state: what client owns after masking
    current_owned = y_true_original[pos] - y_true_masked[pos]
    
    # Only recommend products NOT currently owned
    can_recommend = (current_owned == 0)
    
    # Apply per-product thresholds only to recommendable products
    recommendations = np.zeros(len(scores), dtype=np.uint8)
    for j in range(len(scores)):
        if can_recommend[j] and scores[j] >= per_product_thresholds[j]:
            recommendations[j] = 1
    
    # Evaluation metrics
    # True Positives: recommended AND was masked (omitted)
    tp = np.sum(recommendations & y_true_masked[pos])
    
    # Recall: what fraction of masked products did we recommend?
    masked_count = y_true_masked[pos].sum()
    recall = tp / masked_count if masked_count > 0 else 1.0
    
    # Precision: of all recommendations, how many were actually masked?
    rec_count = recommendations.sum()
    precision = tp / rec_count if rec_count > 0 else (1.0 if masked_count == 0 else 0.0)
    
    return recall, precision, recommendations


def compute_per_product_metrics(
    prod_mat_query, prod_mat_base, y_true_original, y_true_masked,
    index_annoy, per_product_thresholds, nb_similar, products_list
):
    """
    Compute false positive and false negative rates per product.
    """
    num_products = len(products_list)
    false_positives = np.zeros(num_products, dtype=int)
    false_negatives = np.zeros(num_products, dtype=int)
    actual_positives = np.zeros(num_products, dtype=int)
    actual_negatives = np.zeros(num_products, dtype=int)
    
    for pos in range(len(prod_mat_query)):
        vec = np.asarray(prod_mat_query[pos], dtype=np.float32)
        scores = compute_weighted_scores(index_annoy, vec, nb_similar, prod_mat_base)
        
        current_owned = y_true_original[pos] - y_true_masked[pos]
        can_recommend = (current_owned == 0)
        
        recommendations = np.zeros(num_products, dtype=np.uint8)
        for j in range(num_products):
            if can_recommend[j] and scores[j] >= per_product_thresholds[j]:
                recommendations[j] = 1
        
        actual = y_true_masked[pos]
        
        for j in range(num_products):
            # Only count metrics for recommendable products
            if can_recommend[j]:
                if actual[j] == 1:
                    actual_positives[j] += 1
                    if recommendations[j] == 0:
                        false_negatives[j] += 1
                else:
                    actual_negatives[j] += 1
                    if recommendations[j] == 1:
                        false_positives[j] += 1
    
    return false_positives, false_negatives, actual_positives, actual_negatives


if __name__ == "__main__":
    produits_clients_path = "final_data_13_products.csv"
    produits_clients_df = pd.read_csv(produits_clients_path)
    
    trial_recalls = []
    trial_precisions = []
    trial_f1s = []
    alpha = 1.0  # Fraction of clients to modify
    num_trials = 1  # Set to 1 for faster testing, increase for robust results
    nb_similar = 100
    
    for trial in range(num_trials):
        print(f"\n{'='*50}")
        print(f"TRIAL {trial + 1}/{num_trials}")
        print(f"{'='*50}")
        
        # Split data
        index_df, test_df = train_test_split(
            produits_clients_df, test_size=0.1, random_state=42 + trial
        )
        
        # Build Annoy index
        nb_cols = index_df.shape[1]
        vectors_index = np.ascontiguousarray(index_df.to_numpy(dtype=np.float32))
        index_annoy = AnnoyIndex(nb_cols, "manhattan")
        for i, v in enumerate(vectors_index):
            index_annoy.add_item(i, v)
        index_annoy.build(100)
        
        products_list = list(index_df.columns[4:])
        prod_mat_index = np.ascontiguousarray(
            index_df[products_list].to_numpy(dtype=np.float32)
        )
        
        # Original ground truth
        y_true_original = test_df[products_list].to_numpy(dtype=np.uint8)
        
        # Mask some products
        dataset_test, mask = reset_half_of_positives(
            test_df, products_list, alpha=alpha
        )
        y_true_masked = mask.astype(np.uint8)
        
        dataset_test = dataset_test.reset_index(drop=True)
        prod_mat_test = np.ascontiguousarray(
            dataset_test[products_list].to_numpy(dtype=np.float32)
        )
        prod_mat_test_full = np.ascontiguousarray(
            dataset_test.to_numpy(dtype=np.float32)
        )
        
        # Estimate per-product thresholds
        per_product_thresholds = estimate_per_product_thresholds(
            prod_mat_test_full,
            prod_mat_index,
            y_true_original,
            y_true_masked,
            index_annoy,
            nb_similar,
            products_list
        )
        
        print(f"\nPer-product thresholds: {np.round(per_product_thresholds, 3)}")
        print(f"Mean threshold: {np.mean(per_product_thresholds):.3f}")
        
        # Evaluate
        print("\nEvaluating recommendations...")
        with ThreadPoolExecutor(max_workers=None) as ex:
            eval_one_partial = partial(
                eval_one_corrected,
                X_prod_base=prod_mat_index,
                y_true_original=y_true_original,
                y_true_masked=y_true_masked,
                index_annoy=index_annoy,
                per_product_thresholds=per_product_thresholds,
                nb_similar=nb_similar,
                query_vectors=prod_mat_test_full,
            )
            
            results = list(
                tqdm(
                    ex.map(eval_one_partial, range(len(dataset_test))),
                    total=len(dataset_test),
                    desc=f"Trial {trial + 1} - Evaluation"
                )
            )
        
        recalls = np.array([r for r, p, _ in results], dtype=np.float32)
        precisions = np.array([p for r, p, _ in results], dtype=np.float32)
        
        mean_recall = recalls.mean()
        mean_precision = precisions.mean()
        f1 = (
            2 * mean_recall * mean_precision / (mean_recall + mean_precision + 1e-8)
        )
        
        trial_recalls.append(mean_recall)
        trial_precisions.append(mean_precision)
        trial_f1s.append(f1)
        
        print(f"\nTrial {trial + 1} Results:")
        print(f"  Recall:    {mean_recall:.4f}")
        print(f"  Precision: {mean_precision:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        # Compute per-product metrics
        if trial == num_trials - 1:  # Only for last trial
            print("\nComputing per-product metrics...")
            fp, fn, ap, an = compute_per_product_metrics(
                prod_mat_test_full,
                prod_mat_index,
                y_true_original,
                y_true_masked,
                index_annoy,
                per_product_thresholds,
                nb_similar,
                products_list
            )
            
            print("\n" + "="*80)
            print("PER-PRODUCT ANALYSIS (Last Trial)")
            print("="*80)
            for j, product in enumerate(products_list):
                fp_rate = fp[j] / an[j] if an[j] > 0 else 0.0
                fn_rate = fn[j] / ap[j] if ap[j] > 0 else 0.0
                
                print(f"\nProduct: {product}")
                print(f"  Threshold: {per_product_thresholds[j]:.3f}")
                print(f"  False Positive Rate: {fp_rate:.3f} ({fp[j]}/{an[j]})")
                print(f"  False Negative Rate: {fn_rate:.3f} ({fn[j]}/{ap[j]})")
                print(f"  Missed Recommendations: {fn[j]} out of {ap[j]} opportunities")
    
    print("\n" + "="*80)
    print("OVERALL RESULTS ACROSS ALL TRIALS")
    print("="*80)
    print(f"Mean Recall:     {np.mean(trial_recalls):.4f} ± {np.std(trial_recalls):.4f}")
    print(f"Mean Precision:  {np.mean(trial_precisions):.4f} ± {np.std(trial_precisions):.4f}")
    print(f"Mean F1 Score:   {np.mean(trial_f1s):.4f} ± {np.std(trial_f1s):.4f}")
    print("="*80)
