#!/usr/bin/python3
"""
Code to help judge the success of a retrieval run.
"""
import argparse

from ir_eval import BatchSearchResults, Qrels


def get_args():
    options = argparse.ArgumentParser(description='measure success of a run at improving over a baseline')
    options.add_argument('-q', '--qrels', help='qrels file', required=True)
    options.add_argument('-r', '--run', help='experiment run file in TREC output format', required=True)
    options.add_argument('-b', '--baseline', help='baseline run file in TREC output format', required=True)
    args = options.parse_args()
    return args


def get_changes(qrels, baseline, experiment):
    print('query,document,baseline,experiment,change')
    for query in experiment.results:
        for experiment_rank, hit in enumerate(experiment.results[query]):
            if qrels.relevant(query, hit.docno):
                baseline_rank = baseline.rank_of(hit.docno, query)-1  # subtract 1 to make them equivalent
                change = baseline_rank - experiment_rank

                # increment both ranks to make them start at 1
                print(query, hit.docno, baseline_rank+1, experiment_rank+1, change, sep=',')


def main():
    args = get_args()

    qrels = Qrels()
    qrels.read(args.qrels)

    baseline = BatchSearchResults()
    baseline.read(args.baseline)

    experiment = BatchSearchResults()
    experiment.read(args.run)

    get_changes(qrels, baseline, experiment)


if __name__ == '__main__':
    main()
