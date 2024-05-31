import argparse
import os
import tqdm
import pickle
import multiprocessing as mp
from datasets import load_dataset
from rouge_score import rouge_scorer
from datasketch import MinHash, MinHashLSH
from utils import update_args
from src.pipeline.utils import push_to_hf_hub

find_duplicates = None
filter_function = None
apply_filter = None
pool_map_with_progress = None

# Define the function to handle CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Find duplicate messages in a dataset using MinHashLSH.")
    parser.add_argument("--dataset", type=str, help="The dataset to load.")
    parser.add_argument("--split", type=str, help="The split of the dataset to load.")
    parser.add_argument("--column", type=str, help="The column to aim (e.g., 'messages').")
    parser.add_argument("--num-cpus", type=int, default=mp.cpu_count(), help="Number of CPUs to use (default is full).")
    parser.add_argument("--minhash-threshold", type=float, help="MinHashLSH threshold.")
    parser.add_argument("--num-perm", type=int, default=128, help="Number of permutations for MinHash.")
    parser.add_argument("--score-threshold", type=float, help="Score threshold for duplicates.")

    parser.add_argument("--use-cached-minhash", action="store_true", default=True)
    parser.add_argument("--cached-minhash-path", type=str, default="/tmp/minhashes.pkl")

    parser.add_argument("--use-cached-dedup", action="store_true", default=True)
    parser.add_argument("--cached-dedup-path", type=str, default="/tmp/duplicates.pkl")

    parser.add_argument("--push-to-hf-hub", action="store_true", default=True)
    parser.add_argument("--from-config", type=str, default="config/dedup.yaml", help="set CLI options from YAML config")

    args = parser.parse_args()
    args = update_args(parser, args)

    return args

def convert_to_string(data):
    for item in data:
        if item['role'] == 'assistant':
            return item['content']

def create_minhash(text, num_perm):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

def insert_minhash_to_lsh(idx, message, num_perm):
    message_str = convert_to_string(message)
    minhash = create_minhash(message_str, num_perm)
    return idx, minhash

def find_duplicates(idx, minhash, messages, scorer, lsh, score_threshold):
    duplicates = set()
    for duplicate_idx in lsh.query(minhash):
        if duplicate_idx != idx:
            message_i = convert_to_string(messages[idx])
            message_j = convert_to_string(messages[duplicate_idx])
            score = scorer.score(message_i, message_j)["rougeL"].fmeasure
            if score > score_threshold:
                duplicates.add(duplicate_idx)
    return duplicates

def save_intermediate_results(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_intermediate_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main(args):
    global find_duplicates
    global filter_function
    global apply_filter
    global pool_map_with_progress

    # Load the dataset
    dataset = load_dataset(args.dataset)
    messages = dataset[args.split][args.column]

    # Create a Rouge scorer (outside the function for efficiency)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Build LSH index
    lsh = MinHashLSH(threshold=args.minhash_threshold, num_perm=args.num_perm)

    minhashes_file = args.cached_minhash_path
    duplicates_file = args.cached_dedup_path

    if args.use_cached_minhash:
        if os.path.exists(minhashes_file):
            print(f"Loading MinHashes from {minhashes_file}")
            minhashes = load_intermediate_results(minhashes_file)
        else:
            args.use_cached_minhash = False

    if not args.use_cached_minhash:
        print("Creating MinHashes")
        # Insert MinHashes into LSH using multiprocessing
        with mp.Pool(args.num_cpus) as pool:
            insert_tasks = [(idx, message, args.num_perm) for idx, message in enumerate(messages)]
            minhashes = list(tqdm.tqdm(pool.starmap(insert_minhash_to_lsh, insert_tasks), total=len(insert_tasks), desc="Creating MinHashes"))

        save_intermediate_results(minhashes, minhashes_file)

    if args.use_cached_dedup:
        if os.path.exists(duplicates_file):
            print(f"Loading duplicates from {duplicates_file}")
            all_duplicates = load_intermediate_results(duplicates_file)
        else:
            args.use_cached_dedup = False

    if not args.use_cached_dedup:
        print("Inserting MinHashes into LSH")
        for idx, minhash in tqdm.tqdm(minhashes, desc="Inserting MinHashes"):
            lsh.insert(idx, minhash)

        # Function to find duplicates
        def find_duplicates(idx, minhash):
            duplicates = set()
            for duplicate_idx in lsh.query(minhash):
                if duplicate_idx != idx:
                    message_i = convert_to_string(messages[idx])
                    message_j = convert_to_string(messages[duplicate_idx])
                    score = scorer.score(message_i, message_j)["rougeL"].fmeasure
                    if score > args.score_threshold:
                        duplicates.add(duplicate_idx)
            return duplicates

        # Find duplicates using multiprocessing
        all_duplicates = set()
        with mp.Pool(args.num_cpus) as pool:
            async_results = []
            for idx, minhash in tqdm.tqdm(minhashes, desc="Scheduling Tasks"):
                async_result = pool.apply_async(find_duplicates, args=(idx, minhash))
                async_results.append(async_result)

            for async_result in tqdm.tqdm(async_results, desc="Processing Results"):
                duplicates = async_result.get()
                all_duplicates.update(duplicates)

    
        save_intermediate_results(all_duplicates, duplicates_file)

    # Print or process the duplicates as needed
    print(f"Found {len(all_duplicates)} duplicate messages.")

    # Filter out duplicates based on their indices
    filtered_messages = [
        msg for i, msg in enumerate(messages) if i not in all_duplicates
    ]
    print(filtered_messages[0])

    def filter_function(example):
        # Your filtering logic here
        return example["messages"] in filtered_messages

    def apply_filter(example):
        return filter_function(example)

    # Wrap the pool.map with tqdm for progress bar
    def pool_map_with_progress(pool, func, data):
        # Initialize tqdm
        with tqdm.tqdm(total=len(data)) as pbar:
            results = []
            for result in pool.imap_unordered(func, data):
                results.append(result)
                pbar.update()
            return results

    # Create new filtered dataset
    with mp.Pool(args.num_cpus) as pool:
        filtered_list = pool_map_with_progress(pool, apply_filter, dataset[args.split])

    # Convert back to Hugging Face dataset if needed
    # Add tqdm for filtered_indices
    filtered_indices = []
    with tqdm.tqdm(total=len(filtered_list)) as pbar:
        for i, keep in enumerate(filtered_list):
            if keep:
                filtered_indices.append(i)
            pbar.update()
    filtered_dataset = dataset[args.split].select(filtered_indices)

    print(f"Number of messages filtered out: {len(all_duplicates)}")
    print("Filtered dataset size:", len(filtered_dataset))

    if args.push_to_hf_hub:
        print("Pushing the result to Hugging Face Hub")
        dataset[args.split] = filtered_dataset
        dataset.push_to_hub(args.dataset)

if __name__ == "__main__":
    args = parse_args()
    main(args)
