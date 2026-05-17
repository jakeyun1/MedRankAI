"""
run_benchmark.py

This file executes the testbench.
"""

from scripts.tests import *

def run_benchmark(dataset_name, embeddings, metadata_df, image_paths, id_col, label_col):
    """
    Runs the testbench and returns the results for all adapters (for a given model on a given dataset).

    Args:
        dataset_name : The name of the given dataset
        embeddings : The embeddings computed by the model
        metadata_df : DataFrame object used for mapping an image to its associated diagnosis
        image_paths : The local paths of all images
        id_col : The name of the column that identifies each unique image
        label_col : The name of the column that contains the diagnosis for each image  
    
    Returns:
        results : A map containing the formatted results (JSON-ready) for each adapter
    """
    # MLP
    mlp_summary = MLP_cv(dataset_name, embeddings, metadata_df,
                         image_paths, id_col = id_col, label_col = label_col, n_splits = 5)
    print(f"Completed MLP CV benchmark on {dataset_name}.\n")

    # KNN
    knn_summary = KNN_cv(
        dataset_name, embeddings, metadata_df, image_paths,
        id_col = id_col, label_col = label_col, n_splits = 5
    )

    print(f"Completed KNN CV benchmark on {dataset_name}.\n")

    # LR
    logreg_summary = logistic_regression_cv(dataset_name, embeddings, metadata_df, image_paths, id_col = id_col,
                                     label_col = label_col, n_splits = 5)
    print(f"Completed Logistic Regression CV benchmarks on {dataset_name}.\n")

    # Retrieval
    ret_results = retrieval_eval(dataset_name, embeddings, metadata_df, image_paths,
                     id_col = id_col, label_col = label_col, ks = (1,5,10), per_class=True)
    print(f"Completed retrieval evaluation on {dataset_name}.\n")

    # Clustering
    clustering_results = clustering_eval(
    dataset_name, embeddings, metadata_df, image_paths,
    id_col = id_col, label_col = label_col,
    k_range = range(2, 12)
    )
    print(f"Completed clustering evaluation on {dataset_name}.\n\n")

    # Compile the results
    results = {
        "mlp_cv": mlp_summary,
        "knn_cv": knn_summary,
        "logreg_cv": logreg_summary,
        "retrieval": ret_results,
        "clustering": clustering_results
    }

    # Results are ready to be formatted into a JSON file (json.dump)
    return results