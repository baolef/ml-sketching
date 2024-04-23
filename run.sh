# clustering rna2vec
python scripts/clustering.py -i inputs/SILVA_138_3_8_sliding_0_rna2vec.npz -c 1 -d euclidean
python scripts/clustering.py -i inputs/SILVA_138_3_8_sliding_0_rna2vec.npz -c 2 -d euclidean
python scripts/clustering.py -i inputs/SILVA_138_3_8_sliding_0_rna2vec.npz -c 3 -d euclidean
python scripts/clustering.py -i inputs/SILVA_138_3_8_sliding_0_rna2vec.npz -c 4 -d euclidean
python scripts/clustering.py -i inputs/SILVA_138_3_8_sliding_0_rna2vec.npz -c 5 -d euclidean

# clustering minhash
python scripts/clustering.py -i inputs/SILVA_138_3_8_sliding_0_minhash.npz -c 1 -d hamming
python scripts/clustering.py -i inputs/SILVA_138_3_8_sliding_0_minhash.npz -c 2 -d hamming
python scripts/clustering.py -i inputs/SILVA_138_3_8_sliding_0_minhash.npz -c 3 -d hamming
python scripts/clustering.py -i inputs/SILVA_138_3_8_sliding_0_minhash.npz -c 4 -d hamming
python scripts/clustering.py -i inputs/SILVA_138_3_8_sliding_0_minhash.npz -c 5 -d hamming

# classification rna2vec
python scripts/classification.py -i inputs/SILVA_138_3_8_sliding_0_rna2vec.npz -c 1
python scripts/classification.py -i inputs/SILVA_138_3_8_sliding_0_rna2vec.npz -c 2
python scripts/classification.py -i inputs/SILVA_138_3_8_sliding_0_rna2vec.npz -c 3
python scripts/classification.py -i inputs/SILVA_138_3_8_sliding_0_rna2vec.npz -c 4
python scripts/classification.py -i inputs/SILVA_138_3_8_sliding_0_rna2vec.npz -c 5

# classification minhash
python scripts/classification.py -i inputs/SILVA_138_3_8_sliding_0_minhash.npz -c 1
python scripts/classification.py -i inputs/SILVA_138_3_8_sliding_0_minhash.npz -c 2
python scripts/classification.py -i inputs/SILVA_138_3_8_sliding_0_minhash.npz -c 3
python scripts/classification.py -i inputs/SILVA_138_3_8_sliding_0_minhash.npz -c 4
python scripts/classification.py -i inputs/SILVA_138_3_8_sliding_0_minhash.npz -c 5
