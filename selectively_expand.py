import argparse

from ir_eval import BatchSearchResults


def main():
    options = argparse.ArgumentParser(description='Use predictions to selectively expand')
    options.add_argument('baseline')
    options.add_argument('expansion')
    options.add_argument('predictions')
    args = options.parse_args()

    baseline_run = BatchSearchResults()
    baseline_run.read(args.baseline)

    expansion_run = BatchSearchResults()
    expansion_run.read(args.expansion)

    predictions = {}
    with open(args.predictions) as f:
        for line in f:
            docno, query, prediction = line.strip().split()
            predictions[(docno, query)] = int(prediction)

    for query in baseline_run.results:
        for doc in baseline_run.results[query]:
            prediction = predictions[(doc.docno, query)]
            doc.score = doc.score if prediction == -1 else expansion_run.score_of(doc.docno, query)
        baseline_run.rank(query)

    print(baseline_run)


if __name__ == '__main__':
    main()