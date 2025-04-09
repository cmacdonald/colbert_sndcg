import torch


def tensorize_triples(query_tokenizer, doc_tokenizer, queries, positives, negatives, bsize):
    assert len(queries) == len(positives) == len(negatives)
    assert bsize is None or len(queries) % bsize == 0

    N = len(queries)
    # TODO: adjust for offsets
    Q_ids, Q_mask, Q_offsets = query_tokenizer.tensorize(queries)
    D_ids, D_mask, D_offsets = doc_tokenizer.tensorize(positives + negatives)
    D_ids, D_mask, D_offsets = D_ids.view(2, N, -1), D_mask.view(2, N, -1), D_offsets.view(2,N,-1,2)

    # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    maxlens = D_mask.sum(-1).max(0).values

    # Sort by maxlens
    indices = maxlens.sort().indices
    Q_ids, Q_mask, Q_offsets = Q_ids[indices], Q_mask[indices], Q_offsets[indices]
    D_ids, D_mask, D_offsets = D_ids[:, indices], D_mask[:, indices], D_offsets[:,indices]

    (positive_ids, negative_ids), (positive_mask, negative_mask), (positive_offsets, negative_offsets) = D_ids, D_mask, D_offsets

    query_batches = _split_into_batches(Q_ids, Q_mask, Q_offsets, bsize)
    positive_batches = _split_into_batches(positive_ids, positive_mask, positive_offsets, bsize)
    negative_batches = _split_into_batches(negative_ids, negative_mask, negative_offsets, bsize)

    batches = []
    for (q_ids, q_mask, q_offsets), (p_ids, p_mask, p_offsets), (n_ids, n_mask, n_offsets) in zip(query_batches, positive_batches, negative_batches):
        Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)), torch.cat((q_offsets, q_offsets)))
        D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)), torch.cat((p_offsets, n_offsets)))
        batches.append((Q, D))

    return batches


def _sort_by_length(ids, mask, offsets, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, offsets, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], offsets[indices], reverse_indices


def _split_into_batches(ids, mask, offsets, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize], offsets[offset:offset+bsize]))

    return batches
