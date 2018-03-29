#!/usr/bin/python

import argparse
import collections
import logging
import math


class Qrels(object):
    def __init__(self):
        self.qrels = collections.defaultdict(dict)

    def relevance(self, query, document):
        """
        Get the relevance grade of a document for a query
        :param query: The query name
        :param document: The document name
        :return: The relevance grade (number) of the document for the query
        """
        return (self.qrels[query][document]
                if query in self.qrels and document in self.qrels[query] else 0)

    def relevant(self, query, document):
        """
        Get whether the document is relevant to the query
        :param query: The query name
        :param document: The document name
        :return: A boolean of whether the document is relevant to the query
        """
        return self.relevance(query, document) > 0

    def judged(self, query, document):
        """
        Get whether the document is judged for the query
        :param query: The query name
        :param document: The document name
        :return: A boolean of whether the document is judged for the query
        """
        return query in self.qrels and document in self.qrels[query]

    def num_relevant(self, query):
        """
        Get the number of relevant documents to the query
        :param query: The query name
        :return: The number of relevant documents
        """
        return len(self.relevant_documents(query))

    def relevant_documents(self, query):
        """
        Get a list of documents relevant to the query
        :param query: The query name
        :return: A list of relevant documents
        """
        return [doc for doc in self.qrels[query] if self.relevant(query, doc)]

    def read(self, file):
        """
        Read a qrels file and store the data
        :param file: The file path
        """
        try:
            with open(file) as f:
                for line in f:
                    parts = line.split()
                    try:
                        query, document, score = parts[0], parts[2], float(parts[3])
                        self.qrels[query][document] = score
                    except IndexError:
                        logging.warning('This line does not appear to be from a valid qrels file:')
                        logging.warning(line)
                    except ValueError:
                        logging.warning('The score field ({}) does not appear to be a valid number.'.format(score))
        except IOError:
            logging.error('File {} could not be read.'.format(file))


class SearchResult(object):
    def __init__(self, docno, score):
        self.docno = docno
        self.score = score


class BatchSearchResults(object):
    def __init__(self):
        self.results = collections.defaultdict(list)

    def add(self, query, document, score):
        """
        Add a <query, document, score> tuple and optionally resort the documents for this query
        :param query: The query name
        :param document: The document name
        :param score: The retrieval score of the document for the query
        """
        self.results[query].append(SearchResult(document, score))

    def rank(self, query):
        """
        Sort the documents in descending order by their retrieval score
        :param query: The query to sort
        """
        if query not in self.results:
            logging.warning('Asked to rank documents in query {}, but this query is not present in the batch '
                            'results.'.format(query))
            return
        self.results[query].sort(key=lambda d: d.score, reverse=True)

    def rank_of(self, document, query):
        """
        Get the rank of the document for the query
        :param document: The docno string
        :param query: The query title
        :return: The rank of the document for the query, or -1 if document was not returned for the query
        """
        for i, result in enumerate(self.results[query]):
            if result.docno == document:
                return i+1
        return -1

    def read(self, file):
        """
        Read a TREC output file and store the data
        :param file: The file path
        """
        try:
            with open(file) as f:
                for line in f:
                    parts = line.split()
                    try:
                        query, document, score = parts[0], parts[2], float(parts[4])
                        self.add(query, document, score)
                    except IndexError:
                        logging.warning('This line does not appear to be from a TREC output file:')
                        logging.warning(line)
                    except ValueError:
                        logging.warning('The score field ({}) does not appear to be a valid number.'.format(score))
        except IOError:
            logging.error('File {} could not be read.'.format(file))

    def score_of(self, document, query):
        for search_result in self.results[query]:
            if search_result.docno == document:
                return search_result.score
        return None

    def __str__(self):
        lines = []
        for query in self.results:
            for i, doc in enumerate(self.results[query]):
                lines.append(' '.join((str(query), 'Q0', doc.docno, str(i+1), str(doc.score), 'python-dump')))
        return '\n'.join(lines)


"""MAP"""


def average_precision(query, search_results, qrels, rank_cutoff=None):
    if rank_cutoff:
        search_results = search_results[:rank_cutoff]

    ap = 0.0
    rels = 0

    for i, search_result in enumerate(search_results):
        if qrels.relevant(query, search_result.docno):
            rels += 1
            ap += rels / float(i+1)

    try:
        ap /= qrels.num_relevant(query)
    except ZeroDivisionError:
        ap = 0.0

    return ap


def mean_average_precision(batch_search_results, qrels, rank_cutoff=None):
    map = 0.0
    if batch_search_results.results:
        for query in batch_search_results.results:
            map += average_precision(query, batch_search_results.results[query], qrels, rank_cutoff)
        map /= len(batch_search_results.results)
    return map


# Easier name
map = mean_average_precision


"""nDCG"""

def _dcg_at_rank(relevance, rank):
    return


def discounted_cumulative_gain(query, search_results, qrels, rank_cutoff=None):
    if rank_cutoff:
        search_results = search_results[:rank_cutoff]

    dcg = 0.0
    for i, search_result in enumerate(search_results):
        dcg += (math.pow(2, float(qrels.relevance(query, search_result.docno)))-1) / math.log(float(i)+2)
    return dcg


# Easier name
dcg = discounted_cumulative_gain


def ideal_discounted_cumulative_gain(query, qrels, rank_cutoff=None):
    rels = qrels.qrels.get(query, {})
    docs_in_order = sorted(rels, key=rels.get, reverse=True)
    ideal_results = [SearchResult(doc, i) for i, doc in enumerate(docs_in_order)]
    return discounted_cumulative_gain(query, ideal_results, qrels, rank_cutoff)


# Easier name
idcg = ideal_discounted_cumulative_gain


def normalized_discounted_cumulative_gain(query, search_results, qrels, rank_cutoff=None):
    dcg = discounted_cumulative_gain(query, search_results, qrels, rank_cutoff)
    idcg = ideal_discounted_cumulative_gain(query, qrels, rank_cutoff)
    if idcg == 0:
        return 0
    return dcg / idcg


# Easier name
ndcg = normalized_discounted_cumulative_gain


def average_normalized_discounted_cumulative_gain(batch_search_results, qrels, rank_cutoff=None):
    avg = 0.0
    if batch_search_results.results:
        for query in batch_search_results.results:
            avg += normalized_discounted_cumulative_gain(query, batch_search_results.results[query], qrels, rank_cutoff)
        avg /= len(batch_search_results.results)
    return avg


# Easier name
average_ndcg = average_normalized_discounted_cumulative_gain


def main():
    _map = 'map'
    _ndcg = 'ndcg'

    options = argparse.ArgumentParser(description='Evaluate search results')
    required = options.add_argument_group('required arguments')
    required.add_argument('-q', '--qrels', help='qrels file', required=True)
    required.add_argument('-r', '--results', help='search results file', required=True)
    required.add_argument('-m', '--metric', choices=[_map, _ndcg], help='evaluation metric', required=True)
    options.add_argument('-t', '--topic', help='specific topic to evaluate')
    options.add_argument('-c', '--cutoff', type=int, help='rank cutoff')
    args = options.parse_args()

    qrels = Qrels()
    qrels.read(args.qrels)

    results = BatchSearchResults()
    results.read(args.results)

    if args.topic:
        if args.topic not in results.results:
            print('Topic {} not present in results'.format(args.topic))
            exit()

        if args.metric == _map:
            print('ap: {}'.format(str(average_precision(args.topic, results.results[args.topic],
                                                                       qrels))))
        elif args.metric == _ndcg:
            print('nDCG: {}'.format(str(normalized_discounted_cumulative_gain(args.topic, results.results[args.topic],
                                                                              qrels, args.cutoff))))
    else:
        if args.metric == _map:
            print('map: {}'.format(str(mean_average_precision(results, qrels))))
        elif args.metric == _ndcg:
            print('nDCG: {}'.format(str(average_normalized_discounted_cumulative_gain(results, qrels, args.cutoff))))

if __name__ == '__main__':
    main()
