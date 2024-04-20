# Created by Baole Fang at 10/30/23

import argparse
import os

import numpy as np
from Bio import SeqIO

from rna2vec.generators import DisjointKmerFragmenter, SlidingKmerFragmenter, SeqFragmenter
from tqdm import tqdm

import multiprocessing
from datasketch import MinHash


class InvalidArgException(Exception):
    pass


def minhash(seq):
    m=MinHash(num_perm=args.vec_dim)
    seq = ''.join(splitter.get_acgt_seqs(str(seq)))
    fragments = fragmenter.apply(rng, seq)
    for fragment in fragments:
        m.update(fragment.encode('utf-8'))
    return m.hashvalues


def main():
    seqs = [record.seq for record in records]
    Y = [record.description.split()[1] for record in records]
    with multiprocessing.Pool(os.cpu_count()) as p:
        X = list(tqdm(p.imap(minhash, seqs), total=len(seqs)))
    np.savez(path, X=X, Y=Y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert dna dataset to vectors')
    parser.add_argument('-v', '--vec-dim', help='vector dimension', type=int, default=100)
    parser.add_argument('--hash', help='hash functions', type=int, default=100)
    parser.add_argument('-l', '--low', help='lower bound of k', type=int, default=3)
    parser.add_argument('-u', '--up', help='upper bound of k', type=int, default=8)
    parser.add_argument('-i', '--input', help='path to the input dataset', type=str, required=True)
    parser.add_argument('-t', '--type', help='type of the dataset', type=str, default='fasta')
    parser.add_argument('-f', '--fragment', help='style to fragment the sequence: disjoint or sliding',
                        choices=['disjoint', 'sliding'], default='sliding')
    parser.add_argument('-o', '--output', help='output path', type=str, default='inputs')
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    args = parser.parse_args()

    if args.fragment == 'disjoint':
        fragmenter = DisjointKmerFragmenter(args.low, args.up)
    elif args.fragment == 'sliding':
        fragmenter = SlidingKmerFragmenter(args.low, args.up)
    else:
        raise InvalidArgException('Invalid kmer fragmenter: {}'.format(args.kmer_fragmenter))

    splitter = SeqFragmenter()

    records = list(SeqIO.parse(args.input, args.type))
    rng = np.random.RandomState(args.seed)
    name = os.path.basename(args.input).split('.')[0]
    path = os.path.join(args.output, f'{name}_{args.low}_{args.up}_{args.fragment}_{args.seed}_minhash.npz')
    main()
