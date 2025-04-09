import numpy as np
import torch

def initialise_weights(in_dim, out_dim):
    # initialise the weight matrix with zeroes in the shape
    # out_dim x in_dim (i.e. 128 x 768)
    weights = torch.zeros((out_dim, in_dim))
    # compute how many input dimensions have to be aggregated to one
    # output dimension
    mapping_factor = in_dim//out_dim
    # number of remaining dimensions
    last_dim = in_dim % out_dim
    for i in range(0, out_dim):
        # set the mapping factor as weights to average over (1/mapping_factor) neighbouring input dimensions
        weights[i, i * mapping_factor:(i + 1) * mapping_factor] = (1 / mapping_factor) * torch.ones((1, mapping_factor))

    # check if there are any remaining dimensions
    if last_dim!=0:
        # if yes set weights to average over the remaining dimensions
        weights[out_dim - 1, in_dim - last_dim:] = (1 / last_dim) * torch.ones((1, last_dim))

    return weights

def aggregate_batch(T, offsets, aggregation = 'avg', replication = 'replicate', query=False):
    # aggregation describes how the embeddings will be aggregated and can be either first, last or avg
    # replication_strategy describes how the aggregated embedding will be distributed in the tensor
    # so as to not reduce its dimensionality and can be either 'replicate' or 'pad'. 'replicate' means
    # that the aggregated vector will replace the embeddings of all the original subtokens. 'pad' means
    # that the aggregated vector will be put at the position of one of the original subtokens and the other
    # positions are filled up with 0 embeddings
    # The replication strategy is ignored for documents, where we use 'replicate' by default as it has no influence on the result
    #T is either D or Q, i.e. a batch of tensors of size [query length, 768] or [document length, 768] containing the created embeddings
    #for each of the query/ document tokens
    return torch.stack([aggregate(t,o,aggregation,replication,query) for t,o in zip(T,offsets)])

def aggregate(T, offsets, aggregation = 'avg', replication = 'replicate', query=False):
    T1 = T.clone()

    # array of size len(T) (i.e. query/ doc length), containing True at the
    # positions of tokens beginning with '##' and False at all other positions
    subtoken_mask = np.array(offsets)[:, 0] != 0

    # this will become the aggregated vector
    agg = torch.zeros(T.size()[1])
    # denotes at which index a word that is split into subwords begins
    start_idx = -1
    for i in range(1, subtoken_mask.size):
        # If the current index i is a subtoken (i.e. starting with '##')
        if subtoken_mask[i]:
            # If i is the first subtoken of the word
            if start_idx == -1:
                # The first subword does not start with '##' but is still part of
                # the word, thus save it into agg and set the start index of the
                # word to its position
                agg = T[i - 1]
                if aggregation == 'avg':
                    agg = agg + T[i]
                start_idx = i - 1
                # For the 'avg' aggregation strategy we need to sum up all subword
                # representations, thus add the representation of the current
                # subword to the first one
            else:
                if aggregation == 'avg':
                    agg = agg + T[i]
        else:
            if start_idx != -1:
                # this means that the last tokens were subtokens and now have to be replaced according to the
                #replication strategy
                if aggregation == 'last':
                    agg = T[i - 1]
                elif aggregation == 'avg':
                    # compute average of subword representations
                    agg = agg / (i - start_idx)
                # else agg already contains the representation of the first subword
                # and thus does not have to be altered if the aggregation strategy
                # is 'first'

                if replication == 'pad' and query:
                    # ignore replication strategy for documents, set position of first
                    # subword to contain the aggregated vector
                    T1[start_idx] = agg
                    # pad all other positions with 0-embeddings
                    T1[start_idx + 1:i] = torch.zeros((i - start_idx - 1, T.size()[1]))
                else:
                    # 'replicate': replicate the aggregated vector across all subword
                    # positions
                    T1[start_idx:i] = torch.tile(agg, (i - start_idx, 1))
                #reset start index
                start_idx = -1

    return T1

if __name__=='__main__':
    #test linear layer initialisation
    print(initialise_weights(10, 3))
    print(initialise_weights(12,3))

    #Test subword aggregation
    T = torch.stack((torch.from_numpy(np.arange(1,41).reshape((8,5))),torch.from_numpy(np.arange(1,41).reshape((8,5)))))
    offsets = [[(0,0),(0,2),(3,6),(6,9),(0,1),(0,3),(3,7),(0,4)],[(0,0),(0,2),(3,6),(6,9),(0,1),(0,3),(3,7),(0,4)]]
    #1,2,3 and 5,6 are subtokens
    print(T)
    print(aggregate_batch(T, offsets, 'first', 'replicate', False)) #works
    T = torch.stack((torch.from_numpy(np.arange(1,41).reshape((8,5))),torch.from_numpy(np.arange(1,41).reshape((8,5)))))
    print(aggregate_batch(T, offsets, 'avg', 'replicate', False)) #works
    T = torch.stack((torch.from_numpy(np.arange(1,41).reshape((8,5))),torch.from_numpy(np.arange(1,41).reshape((8,5)))))
    print(aggregate_batch(T, offsets, 'last', 'replicate', False)) #works
    T = torch.stack((torch.from_numpy(np.arange(1,41).reshape((8,5))),torch.from_numpy(np.arange(1,41).reshape((8,5)))))
    print(aggregate_batch(T, offsets, 'first', 'pad', True)) # works
    T = torch.stack((torch.from_numpy(np.arange(1,41).reshape((8,5))),torch.from_numpy(np.arange(1,41).reshape((8,5)))))
    print(aggregate_batch(T, offsets, 'avg', 'pad', True))#works
    T = torch.stack((torch.from_numpy(np.arange(1,41).reshape((8,5))),torch.from_numpy(np.arange(1,41).reshape((8,5)))))
    print(aggregate_batch(T, offsets, 'last', 'pad', True))#works