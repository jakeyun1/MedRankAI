"""
calculate_results.py

Result JSON files are read for computing average metrics per adapter type.
Metrics are broken down both by adapter type and the specific adapter itself.

Metrics are to be used for display on the website.

WARNING: JSON structure is parsed using hardcoded keys. Be weary if editing tests.py.
"""

import numpy as np
import json
import os
import argparse

TASK_MAP = {"pad_ufes": "Skin lesions",
            "odir": "Ocular fundi",
            "ham10000": "Skin lesions",
            "cbis_ddsm": "Mammograms",
            "chexpert": "Chest radiographs"}

METRICS_MAP = {"Classification": ["mlp_cv_f1", "knn_cv_f1", "logreg_cv_f1"],
               "Retrieval": ["recall@5", "map"],
               "Clustering": "nmi"}

def compute_classification_averages(json_list):
    """
    Calculates the average F1 score across the used 
    datasets amongst the MLP, KNN, and LR adapters.

    Sample output:
    {"mlp_cv_f1": [.738, .023], 
    "knn_cv_f1": [.627, .026],
    "logreg_cv_f1": [.364, .0032]}

    results["mlp_cv_f1"][0] is the mean F1 score, results["mlp_cv_f1"][1] is the std. dev.
    of the F1 score

    The F1 scores of the other classification adapters follow this pattern.

    Args:
        json_list : The list of dataset-specific JSON file paths

    Returns:
        classification_results : The computed metrics in a JSON-formatted map
    """
    METRICS = METRICS_MAP["Classification"]

    # Create accumulator for storing computations
    accumulator = [[], [], []]

    # Iterate through the dataset results
    for file in json_list:
        with open(file, "r") as f:
            dataset_results = json.load(f)
        
        # WARNING: accumulator indices are hardcoded due to dataset results JSON structure
        # MLP
        accumulator[0].append(dataset_results["mlp_cv"]["f1_weighted"][0])

        # KNN
        accumulator[1].append(dataset_results["knn_cv"]["best_scores"]["f1_weighted"])

        # LR
        accumulator[2].append(dataset_results["logreg_cv"]["f1_weighted"][0])

    # Compute averages and std devs
    for idx in range(len(accumulator)):
        computed_metrics = [np.mean(accumulator[idx]), np.std(accumulator[idx])]
        accumulator[idx] = computed_metrics

    # Final computed metrics
    classification_results = {METRICS[idx]:f1_scores for idx, f1_scores in enumerate(accumulator)}
    
    return classification_results
        
def compute_retrieval_averages(json_list):
    """
    Calculates the average Recall@5 and average mAP across the
    used datasets.

    Sample output:
    {"recall@5": [.783, .04],
    "map": [.239, .04]}

    results["recall@5"][0] is the mean Recall@5 score, results["recall@5"][1]
    is the std. dev. of the Recall@5 score

    mAP follows this pattern.

    Args:
        json_list : The list of dataset-specific JSON file paths

    Returns:
        retrieval_results : The computed metrics in a JSON-formatted map
    """
    METRICS = METRICS_MAP["Retrieval"]

    # Create accumulator for storing computations
    accumulator = [[], []]

    # Iterate through the dataset results
    for file in json_list:
        with open(file, "r") as f:
            dataset_results = json.load(f)
        
        # WARNING: accumulator indices are hardcoded due to dataset results JSON structure
        # Recall@5
        accumulator[0].append(dataset_results["retrieval"]["recall_at_k"]["5"])

        # mAP
        accumulator[1].append(dataset_results["retrieval"]["map"])

    # Compute averages and std devs
    for idx in range(len(accumulator)):
        computed_metrics = [np.mean(accumulator[idx]), np.std(accumulator[idx])]
        accumulator[idx] = computed_metrics

    # Final computed metrics
    classification_results = {METRICS[idx]:scores for idx, scores in enumerate(accumulator)}
    
    return classification_results

def compute_clustering_averages(json_list):
    """
    Calculates the average NMI across the used datasets.

    Sample output:
    {"nmi": [k_dict, .102, .03]}

    k_dict contains the best k values per task
    
    results["nmi"][1] is the mean NMI score, results["nmi"][2] is the
    std. dev. of the NMI score

    Args:
        json_list : The list of dataset-specific JSON file paths

    Returns:
        clustering_results : The computed metrics in a JSON-formatted map
    """
    METRIC = METRICS_MAP["Clustering"]

    # Create accumulator for storing computations
    accumulator = [{}]

    # Iterate through the dataset results
    for file in json_list:
        with open(file, "r") as f:
            dataset_results = json.load(f)
        
        # WARNING: accumulator indices are hardcoded due to dataset results JSON structure
        # Best k
        dataset = os.path.basename(file)[:-5]
        task = TASK_MAP[dataset]

        accumulator[0][task] = [dataset_results["clustering"]["best_nmi"][0]] if task not in accumulator[0] \
                else accumulator[0][task] + [dataset_results["clustering"]["best_nmi"][0]]

        # NMI
        accumulator.append(dataset_results["clustering"]["best_nmi"][1])

    # Compute average and std dev
    accumulator[0] = {task:int(np.round(np.mean(k_list))) for task, k_list in accumulator[0].items()}
    mean_nmi = np.mean(accumulator[1:])
    std_nmi = np.std(accumulator[1:])
    accumulator = [accumulator[0], mean_nmi, std_nmi]

    # Final computed metrics
    classification_results = {METRIC:accumulator}
    
    return classification_results

# Parse command line arguments
parser = argparse.ArgumentParser(
        prog = "python calculate_results.py",
        description = "Generates a formatted JSON results file for a given model.",
        epilog = "The results argument is optional. If no value is passed then the current directory is used.")
parser.add_argument("--results",
        default = ".",
        help = "Path to a directory that contains the results JSON files for a given model (optional)",
        )
args = parser.parse_args()

# Obtain the results path (directory)
# If no path is passed, the current directory will be used
results_path = args.results

# Test if the given path is valid
if not os.path.isdir(results_path):
    raise NotADirectoryError("The path to the desired directory does not exist or "
            "the path is not a directory.")

# Change the directory of the Python subprocess to the results filepath
os.chdir(results_path)

# Get the name of the model: e.g. 'resnet50'
model_name = os.path.abspath(".").split(os.sep)[-1].split("_")[0]

# Check if the results file exists to prevent overwriting
filename = model_name + "_results.json"
full_path = os.path.join(os.path.abspath("."), filename)

if os.path.exists(full_path):
    raise FileExistsError(f"{full_path} already exists.")

# Grab all files ending in .json excluding the config JSON
curr_dir = os.listdir(".")

json_list = []

for file in curr_dir:
    if file.endswith(".json"):
        if "config" not in file and "results" not in file:
            json_list.append(os.path.abspath(file))

num_datasets = len(json_list)

if num_datasets == 0:
    raise FileNotFoundError("The results directory must contain valid JSON files.")

output = {}

# Obtain results for the types of adapters
classification_results = compute_classification_averages(json_list) # F1 score
retrieval_results = compute_retrieval_averages(json_list) # Recall@5, mAP
clustering_results = compute_clustering_averages(json_list) # NMI

output["Classification"] = classification_results
output["Retrieval"] = retrieval_results
output["Clustering"] = clustering_results

with open(full_path, "w") as f:
    json.dump(output, f, indent = 2)

# Confirmation of creation of results JSON file
message = f"\n{filename} has been created at " + \
        f"{os.sep.join(os.path.dirname(os.path.abspath(filename)).split(os.sep)[-2:])}.\n"
print("=" * len(message.strip()))
print(message)
print("=" * len(message.strip()))