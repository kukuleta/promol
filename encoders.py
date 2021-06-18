import os
import string
from itertools import repeat, product

import pandas as pd

import khmer
import tokenizers
from propy import PyPro
from propy.GetProteinFromUniprot import GetProteinSequence


from rdkit.Chem import AllChem as Chem

#K-Mer Reduction
def extract_kmers(sequence, ksize=2, kfactor=4):
    ksize = ksize
    nkmers = 1e7 # 32 ** ksize
    tablesize = 1 #+ extra_slot_num

    kmer = khmer.Counttable(ksize, 1e7, tablesize)#, tablesize, ))
    kmer.set_use_bigcount(True)
    kmer.consume(sequence)

    return kmer

def get_unique_khmer_set(alphabet, ksize):
    items = sorted(list(map(lambda x: "".join(x), product(*[alphabet \
                                                            for ix in range(ksize)]))))
    return items

def get_kmer_counts(kmers, items):
    return dict(zip(items, map(lambda x: kmers.get(x), items)))


def get_residue_patterns_from_composed_sequence(sequence, items, kmer_size=2):
    kmers = extract_kmers(sequence)
    return pd.Series(get_kmer_counts(kmers, items))


def concat_kmer_patterns(sequence_list, alphabet, kmer_size=2):
    alphabet_letters = sorted(set(alphabet))
    items = get_unique_khmer_set(alphabet_letters, kmer_size)

    return pd.concat([get_residue_patterns_from_composed_sequence(sequence, items, kmer_size) \
                      for sequence in sequence_list], axis=1)

#Protein Composition
def extract_composition(sequence, composition="AAComp"):
    protein_properties = PyPro.GetProDes(sequence)
    compositions = getattr(protein_properties, f"Get{composition}")()
    return pd.Series(compositions.values(), index=pd.Index(compositions.keys(), name=sequence))

def concat_compositions(sequence, compositions):
    return pd.concat([extract_composition(sequence, composition) for composition in compositions]).T

def get_morgan_fingerprint(x):
    try:
        rep = Chem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), radius=2, nBits=2048)

    except:
        return None

    else:
        return rep

def construct_tokenization_vocabulary(model,
                                      files,
                                      trainer_params,
                                      tokenizer_params,
                                      output_file):

    tokenizer = model(**tokenizer_params)
    tokenizer.train(files, **trainer_params)

    tokenizer.save(output_file, pretty=True)

def format_encoded_tokens(sequence, tokens):

    return f"Before tokenization : {sequence} \n\n" \
            + "-" * 20 + "\n\n" \
            + f"Tokens : {tokens}\n\n".format(" ".join(tokens)) \
            + "-" * 20 + "\n\n" \
            + "After tokenization : " + " | ".join(tokens)