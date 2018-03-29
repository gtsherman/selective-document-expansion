from argparse import ArgumentParser
from sys import stderr

from pandas import read_csv
from sklearn import linear_model, naive_bayes, neighbors, dummy
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict, train_test_split


def main():
    options = ArgumentParser(description='Classify')
    options.add_argument('--train', help='training data (requires --test)')
    options.add_argument('--test', help='testing data (requires --train)')
    options.add_argument('--data', help='data for cross validation or train/test split mode (default=CV)')
    options.add_argument('--rank-changes', help='file containing rank change information for query-document pairs')
    options.add_argument('--mode', choices=['cv', 'split'], default='cv', help='cross validation (cv) or train/test '
                                                                               'split (split) mode for evaluating a '
                                                                               'single dataset')
    options.add_argument('--folds', type=int, default=10, help='number of folds for cross-validation; default=10')
    options.add_argument('--test-split', type=float, default=0.25, help='percentage of data to be used as the test '
                                                                        'split')
    args = options.parse_args()

    # Check args are correct
    if (args.train and not args.test) or (args.test and not args.train):
        print('Must specify both --train and --test. Otherwise use --data.', file=stderr)
        quit()
    if args.data and args.train:
        print('--data used with --train. Ignoring --train.', file=stderr)
    if args.data and args.test:
        print('--data used with --test. Ignoring --test.', file=stderr)

    # Read data
    if args.train and args.test:
        with open(args.train) as f:
            training_data = read_csv(f)

        with open(args.test) as f:
            testing_data = read_csv(f)

        if args.rank_changes:
            with open(args.rank_changes) as f:
                rank_changes = read_csv(f)

            training_data = training_data.merge(rank_changes)
            testing_data = testing_data.merge(rank_changes)

        training_data, training_features, training_labels = prepare_data(training_data)
        testing_data, testing_features, testing_labels = prepare_data(testing_data)


    if args.data:
        with open(args.data) as f:
            data = read_csv(f)

        if args.rank_changes:
            with open(args.rank_changes) as f:
                rank_changes = read_csv(f)

            data = data.merge(rank_changes)

        data, features, labels = prepare_data(data)

        if args.mode == 'split':
            training_features, testing_features, training_labels, testing_labels = train_test_split(features, labels,
                                                                                                    args.test_size)
        else:
            run_cv(data, features, labels, args.folds)
            quit()

    baseline = dummy.DummyClassifier(strategy='most_frequent')
    baseline.fit(training_features, training_labels)
    predictions = baseline.predict(testing_features)
    print('Train/test baseline most frequent:', accuracy_score(testing_labels, predictions), file=stderr)

    nb = naive_bayes.GaussianNB()
    nb.fit(training_features, training_labels)
    predictions = nb.predict(testing_features)
    print('Train/test naive Bayes:', accuracy_score(testing_labels, predictions), file=stderr)

    knn_uni = neighbors.KNeighborsClassifier()
    knn_uni.fit(training_features, training_labels)
    predictions = knn_uni.predict(testing_features)
    print('Train/test KNN (uniform):', accuracy_score(testing_labels, predictions), file=stderr)

    knn_dist = neighbors.KNeighborsClassifier(weights='distance')
    knn_dist.fit(training_features, training_labels)
    predictions = knn_dist.predict(testing_features)
    print('Train/test KNN (distance):', accuracy_score(testing_labels, predictions), file=stderr)

    logreg = linear_model.LogisticRegression()
    #rfecv = RFECV(logreg, cv=10)
    logreg.fit(training_features, training_labels)
    predictions = logreg.predict(testing_features)
    print('Train/test logistic regression:', accuracy_score(testing_labels, predictions), file=stderr)

    for i, prediction in enumerate(predictions):
        print(testing_data['docno'][i], testing_data['query'][i], prediction)


def run_cv(data, features, labels, folds):
    baseline = dummy.DummyClassifier(strategy='most_frequent')
    predictions = cross_val_predict(baseline, features, labels, cv=folds, n_jobs=-1)
    print('Cross-validated baseline most frequent:', accuracy_score(labels, predictions), file=stderr)

    nb = naive_bayes.GaussianNB()
    predictions = cross_val_predict(nb, features, labels, cv=folds, n_jobs=-1)
    print('Cross-validated naive Bayes:', accuracy_score(labels, predictions), file=stderr)

    knn_uni = neighbors.KNeighborsClassifier()
    predictions = cross_val_predict(knn_uni, features, labels, cv=folds, n_jobs=-1)
    print('Cross-validated KNN (uniform):', accuracy_score(labels, predictions), file=stderr)

    knn_dist = neighbors.KNeighborsClassifier(weights='distance')
    predictions = cross_val_predict(knn_dist, features, labels, cv=folds, n_jobs=-1)
    print('Cross-validated KNN (distance):', accuracy_score(labels, predictions), file=stderr)

    logreg = linear_model.LogisticRegression()
    # rfecv = RFECV(logreg, cv=10)
    predictions = cross_val_predict(logreg, features, labels, cv=folds, n_jobs=-1)
    print('Cross-validated logistic regression:', accuracy_score(labels, predictions), file=stderr)

    for i, prediction in enumerate(predictions):
        print(data['docno'][i], data['query'][i], prediction)


def filter_features(data):
    return data[data.columns.difference(['docno', 'query', 'relevance', 'rankImproved', 'rankImprovement', 'label',
                                         'rel', 'baseline_rank', 'experiment_rank', 'rank_change', 'baseline_ql',
                                         'experiment_ql', 'ql_change', 'qlImproved', 'originalToExpandedKL',
                                         'originalToExpandedCosine'])]


def prepare_data(data):
    data = convert_to_classes(data)
    labels = data['label'].values
    features = filter_features(data)
    return data, features, labels


def convert_to_classes(data):
    try:
        # if there is no rank_change, the doc's rank dropped below 1000 and therefore decreased (this shouldn't actually
        # happen since we simply re-rank the docs returned for the baseline, but different baselines exist so...)
        data['rank_change'] = data['rank_change'].fillna(value=-1)

        # rankImproved = binary variable of rank improvement (-1 includes those whose rank stayed the same)
        data['rankImproved'] = data['rank_change'].apply(lambda rc: 1 if rc > 0 else -1)

        # if the rank improved (1) and the document is relevant (rel=1), label=1
        # if the rank decreased (-1) and the document is nonrelevant (rel=-1), label=1  -- this is a good result
        # if the rank improvement and relevance do not match, it is not what we want, label=-1
        data['label'] = data['rankImproved'] * _binary_relevance(data['relevance']) + 0

        # data['qlImproved'] = data['ql_change'].apply(lambda qc: 1 if qc > 0 else -1)
        # data['label'] = data['qlImproved'] * _binary_relevance(data['relevance']) + 0
    except KeyError:
        print('Error identifying classes. Did you forget the --rank-changes option?', file=stderr)
        quit()

    return data


def _binary_relevance(raw_relevance):
    return raw_relevance.apply(lambda rel: 1 if rel > 0 else -1)


if __name__ == '__main__':
    main()
