import argparse

from ir_eval import BatchSearchResults, Qrels


def get_args():
    options = argparse.ArgumentParser(description='measure success of a run at improving over a baseline')
    options.add_argument('-q', '--qrels', help='qrels file', required=True)
    options.add_argument('-r', '--run', help='experiment run file in TREC output format', required=True)
    options.add_argument('-b', '--baseline', help='baseline run file in TREC output format', required=True)
    args = options.parse_args()
    return args


def only_nonrel_docs(batch_search_results, query, qrels, judged_only=False, keep=()):
    nonrel_results = []
    for search_result in batch_search_results.results[query]:
        # if it's a nonrel doc or if it's in the list of docs to keep
        if not qrels.relevant(query, search_result.docno) or search_result.docno in keep:
            if judged_only and not qrels.judged(query, search_result.docno):  # if not judged but we only want judged
                continue  # skip it
            nonrel_results.append(search_result.docno)
    return nonrel_results


def main():
    args = get_args()

    qrels = Qrels()
    qrels.read(args.qrels)

    baseline = BatchSearchResults()
    baseline.read(args.baseline)

    experiment = BatchSearchResults()
    experiment.read(args.run)

    print('query,document,baseline,experiment,change')
    for query in qrels.qrels:
        for doc in qrels.qrels[query]:
            try:
                baseline_rank = only_nonrel_docs(baseline, query, qrels, judged_only=True, keep=(doc)).index(doc) + 1
            except ValueError:
                baseline_rank = -1

            try:
                test_rank = only_nonrel_docs(experiment, query, qrels, judged_only=True, keep=(doc)).index(doc) + 1
            except ValueError:
                test_rank = -1

            print(query, doc, baseline_rank, test_rank, baseline_rank - test_rank, sep=',')


if __name__ == '__main__':
    main()
