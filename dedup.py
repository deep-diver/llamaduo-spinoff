import argparse
import tqdm
import multiprocessing as mp
from datasets import load_dataset
from rouge_score import rouge_scorer
from datasketch import MinHash, MinHashLSH

from utils import update_args
from src.pipeline.utils import push_to_hf_hub

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

    parser.add_argument("--push-to-hf-hub", action="store_true", default=True)

    parser.add_argument("--from-config", type=str, default="config/dedup.yaml", help="set CLI options from YAML config")

    args = parser.parse_args()
    args = update_args(parser, args)

    return args

def convert_to_string(data):
    result = []
    for item in data:
        if item['role'] == 'user':
            result.append(f"user: {item['content']}")
        elif item['role'] == 'assistant':
            result.append(f"assistant: {item['content']}")
    return '\n'.join(result)

def create_minhash(text, num_perm):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

def find_duplicates(args, messages, idx, minhash, scorer, lsh):
    duplicates = set()
    for duplicate_idx in lsh.query(minhash):
        if duplicate_idx != idx:
            message_i = convert_to_string(messages[idx])
            message_j = convert_to_string(messages[duplicate_idx])
            score = scorer.score(message_i, message_j)["rougeL"].fmeasure
            if score > args.score_threshold:
                duplicates.add(duplicate_idx)
    return duplicates

def main(args):
    # Load the dataset
    dataset = load_dataset(args.dataset)
    messages = dataset[args.split][args.column]

    # Create a Rouge scorer (outside the function for efficiency)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Define number of processes
    num_processes = args.num_cpus

    # Build LSH index
    lsh = MinHashLSH(threshold=args.minhash_threshold, num_perm=args.num_perm)
    minhashes = []
    for idx, message in enumerate(tqdm.tqdm(messages, desc="Creating MinHashes")):
        message_str = convert_to_string(message)
        minhash = create_minhash(message_str, args.num_perm)
        lsh.insert(idx, minhash)
        minhashes.append((idx, minhash))

    # Find duplicates using multiprocessing
    all_duplicates = set()
    with mp.Pool(num_processes) as pool:
        async_results = []
        for idx, minhash in tqdm.tqdm(minhashes, desc="Scheduling Tasks"):
            async_result = pool.apply_async(find_duplicates, args=(args, messages, idx, minhash, scorer, lsh))
            async_results.append(async_result)

        for async_result in tqdm.tqdm(async_results, desc="Processing Results"):
            duplicates = async_result.get()
            all_duplicates.update(duplicates)

    # Print or process the duplicates as needed
    print(f"Found {len(all_duplicates)} duplicate messages.")

    # Filter out duplicates based on their indices
    filtered_messages = [
        msg for i, msg in enumerate(messages) if i not in all_duplicates
    ]

    # Create new filtered dataset
    filtered_dataset = dataset.filter(lambda example, idx: example["messages"] in filtered_messages, with_indices=True)

    print(f"Number of messages filtered out: {len(all_duplicates)}")
    print("Filtered dataset size:", len(filtered_dataset))

    if args.push_to_hf_hub:
        print("Pushing the result to Hugging Face Hub")
        dataset[args.split] = filtered_dataset
        dataset.push_to_hub(args.dataset)

if __name__ == "__main__":
    args = parse_args()
    main(args)

