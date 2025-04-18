import torch

from transformers import BertTokenizerFast
from colbert.modeling.tokenization.utils import _split_into_batches, _sort_by_length


class DocTokenizer():
    def __init__(self, doc_maxlen):
        self.tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.doc_maxlen = doc_maxlen

        self.D_marker_token, self.D_marker_token_id = '[D]', self.tok.convert_tokens_to_ids('[unused1]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id

        assert self.D_marker_token_id == 2

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.D_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [D] marker
        batch_text = [('. ' + x).split() for x in batch_text]

        obj = self.tok(batch_text, padding='longest', truncation='longest_first', is_split_into_words=True,
                       return_tensors='pt', max_length=self.doc_maxlen, return_offsets_mapping=True)

        #offsets are used for subword aggregation
        #offsets is a list of tuples per sample in the batch, it contains one tuple per token denoting the
        #start and end of the token in the input sequence
        #as the input sequence is pre-split, the start will only be non-zero for subword tokens (i.e. tokens starting with ##)

        ids, mask, offsets = obj['input_ids'], obj['attention_mask'], obj['offset_mapping']

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        if bsize:
            ids, mask, offsets, reverse_indices = _sort_by_length(ids, mask, offsets, bsize)
            batches = _split_into_batches(ids, mask, offsets, bsize)
            return batches, reverse_indices

        return ids, mask, offsets
