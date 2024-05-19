import pdb
import numpy as np


def read_seq_graphprot(seq_file, label):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    return seq_list, labels


def read_data_file(posifile, negafile):
    data = dict()
    seqs1, labels1 = read_seq_graphprot(posifile, label=1)
    seqs2, labels2 = read_seq_graphprot(negafile, label=0)
    seqs = seqs1 + seqs2
    labels = labels1 + labels2
    # print(labels)
    data["seq"] = seqs
    data["Y"] = np.array(labels)
    return data


# If the length is less than 501, fill it with N; If the length is greater than 501, only the first 501 nucleotides
# are taken
def padding_sequence(seq, max_len=501, repkey='N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len - seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq


# Transform processed equal-length (501) RNA sequence into a matrix
def get_RNA_seq_concolutional_array(seq, motif_len=4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len - 1):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - 3, row):
        new_array[i] = np.array([0.25] * 4)

    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    return new_array


def get_bag_data_1_channel(data, max_len):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len=max_len)
        bag_subt = []
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags, labels


# channel
def get_data(posi, nega, window_size):
    # Read positive and negative samples and process to get standardized data and label
    data = read_data_file(posi, nega)
    train_bags, label = get_bag_data_1_channel(data, max_len=window_size)
    return train_bags, label