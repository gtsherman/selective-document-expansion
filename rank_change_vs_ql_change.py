from ir_eval import BatchSearchResults, Qrels
from rank_change_one_doc import get_args, only_nonrel_docs


def main():
    args = get_args()

    qrels = Qrels()
    qrels.read(args.qrels)

    baseline = BatchSearchResults()
    baseline.read(args.baseline)

    experiment = BatchSearchResults()
    experiment.read(args.run)

    print('docno,query,rel,baseline_rank,experiment_rank,rank_change,baseline_ql,experiment_ql,ql_change')

    for query in baseline.results:
        baseline_docs = set([search_result.docno for search_result in baseline.results[query]])
        experiment_docs = set([search_result.docno for search_result in experiment.results[query]])
        #docs = baseline_docs.union(experiment_docs)

        for doc in baseline_docs:
            baseline_rank = baseline.rank_of(doc, query)
            test_rank = experiment.rank_of(doc, query)

            baseline_ql = baseline.score_of(doc, query)
            experiment_ql = experiment.score_of(doc, query)

            try:
                rank_diff = baseline_rank - test_rank
            except TypeError:
                rank_diff = None

            try:
                ql_diff = experiment_ql - baseline_ql
            except TypeError:
                ql_diff = None

            print(doc, query, qrels.relevance(query, doc), baseline_rank, test_rank, rank_diff,
                  baseline_ql, experiment_ql, ql_diff, sep=',')


if __name__ == '__main__':
    main()
