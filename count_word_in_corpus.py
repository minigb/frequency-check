from datasets import load_dataset
import pandas as pd
import os
import argparse
import json
from collections import defaultdict
from tqdm import tqdm
import re
import ahocorasick


def main():
    """
    Usage:
        $ python count_word_in_corpus.py -i <path_to_targets_json_file> -o <path_to_save_output_csv> [-c <path_to_corpus_file>]

    Arguments:
        -i, --input_dir          : Path to the JSON file containing the list of targets.
        -o, --output_file_name   : Path where the result CSV will be saved.
        -c, --corpus_dir         : (Optional) Path to the corpus file. Defaults to 'dataset/music_corpus.jsonl' if not provided.

    Example:
        $ python count_word_in_corpus.py -i targets.json -o results.csv -c music_corpus.jsonl
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json_name", required = True, help = "Path to the directory of the target json file")
    parser.add_argument("-o", "--output_csv_name", required = True, help = "Path to the directory to save the result")
    parser.add_argument("-c", "--corpus_dir", default = None, help = "Path to the corpus file")
    args = parser.parse_args()

    # input dir
    input_json_name = args.input_json_name
    assert os.path.exists(input_json_name), f'input dir {input_json_name} does not exist'
    assert input_json_name.endswith('.json') or input_json_name.endswith('.jsonl'),\
        f'Input file should be json or jsonl'

    # corpus dir
    corpus_dir = args.corpus_dir if args.corpus_dir else 'dataset/music_corpus.jsonl'
    assert os.path.exists(corpus_dir), f'corpus dir {corpus_dir} does not exist'

    # output dir
    output_csv_name = args.output_csv_name
    assert output_csv_name.endswith('.csv'), f'Output file name should end with .csv'

    # Get json file containing target
    # TODO(minigb): This assumes that it is a json file
    print('Reading target list')
    with open(input_json_name, 'r') as file:
        target_list = json.load(file)
    print('Done')

    # Get corpus file
    # TODO(minigb): Remove duplicates
    print('Reading corpus')
    with open(corpus_dir) as f:
        corpus = [json.loads(line) for line in tqdm(f)]
    corpus_text_list = [element['text'] for element in corpus]
    print('Done')

    A = ahocorasick.Automaton()
    for _, target in enumerate(target_list):
        A.add_word(target, target)
    A.make_automaton()

    total_count = {target : 0 for target in target_list}
    n_of_docs = {target : 0 for target in target_list}

    for text in tqdm(corpus_text_list):
        target_count_in_this_text = defaultdict(int)
        for _, target in A.iter(text):
            target_count_in_this_text[target] += 1
        for target, count in target_count_in_this_text.items():
            total_count[target] += count
            n_of_docs[target] += 1

    result_df = pd.DataFrame({
        'target': list(total_count.keys()),
        'total_count': list(total_count.values()),
        'num_of_documents': list(n_of_docs.values())
    })
    
    result_df.sort_values(by = 'total_count', ascending = False, inplace = True)
    result_df.set_index('target', inplace = True)
    result_df.to_csv(args.output_csv_name)

if __name__ == "__main__":
    main()