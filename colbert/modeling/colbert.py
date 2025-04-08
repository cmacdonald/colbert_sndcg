import string
import torch
import torch.nn as nn
import numpy as np

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colbert.parameters import DEVICE
from colbert.modeling.utils import initialise_weights, aggregate_batch


class ColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine', linear_layer='original', aggregation='none', replication='replicate'):

        """
        @param linear_layer: the version of the linear compression layer for the model, can be 'original',
            'none', 'static_mapping', 'trained_mapping' or 'static_random'
        @param aggregation: the subword aggregation strategy (if any) for the model, can be 'first', 'avg', 'last' or 'none'
        @param replication: the replication strategy (in case the aggregation parameter is set to something other than
            'none'), can be either 'replicate' or 'pad'
        """

        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim
        self.aggregation = aggregation
        self.replication = replication
        self.tok = BertTokenizerFast.from_pretrained('bert-base-uncased')

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = BertModel(config)
        print("aggregation",aggregation)
        if linear_layer=='none':
            print('none')
            self.linear = None
        elif linear_layer=='static_mapping':
            #linear layer is initialised with weights that correspond to a simple mapping and combination of values
            #the layer is not trained, i.e. its weights do not change
            print('static_mapping')
            self.linear = nn.Linear(config.hidden_size, dim, bias=False)
            self.linear.weight = torch.nn.Parameter(initialise_weights(self.config.hidden_size, self.dim), requires_grad=False)
            #set linear layer weights to not be trained
            for param in self.linear.parameters():
                param.requires_grad = False
            self.linear._is_hf_initialized = True #This line is VERY important so that the linear layer weights aren't initialised randomly
        elif linear_layer=='trained_mapping':
            print('trained_mapping')
            # linear layer is initialised with weights that correspond to a simple mapping and combination of values
            #this layer is trained, i.e. the weights might change
            self.linear = nn.Linear(config.hidden_size, dim, bias=False)
            self.linear.weight = torch.nn.Parameter(initialise_weights(self.config.hidden_size, self.dim), requires_grad=True)
            self.linear._is_hf_initialized = True #This line is VERY important so that the linear layer weights aren't initialised randomly
        elif linear_layer=='static_random':
            print('static_random')
            self.linear = nn.Linear(config.hidden_size, dim, bias=False)
            #set linear layer weights to not be trained
            for param in self.linear.parameters():
                param.requires_grad = False
        else:
            #'original'
            # learned linear layer as in the original paper
            print('original')
            self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()

    def forward(self, Q, D):
        scores = self.score(self.query(*Q), self.doc(*D))
        return scores

    def query(self, input_ids, attention_mask, offsets, indices=None):
        #produces the query embeddings
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]

        if self.aggregation in ['first', 'last', 'avg']:
            #if a subword aggregation strategy is set, apply it to the query
            Q = aggregate_batch(Q, offsets, self.aggregation, self.replication, query=True)
        if self.linear:
            #apply the linear layer to the query, if there is one
            Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def query_without_linear(self, input_ids, attention_mask):
        #used in ../evaluation/clustering.py to obtain the query embeddings before the linear layer
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        return Q

    def apply_linear(self, T):
        #used in .../evaluation/clustering.py to apply the linear layer to the obtained embeddings without length normalisation
        T = self.linear(T)
        return T

    def doc(self, input_ids, attention_mask, offsets, keep_dims=True):
        #produces the document token embeddings
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]

        if self.aggregation in ['first', 'last', 'avg']:
            #if a subword aggregation strategy is set, apply it to the obtained document embeddings
            D = aggregate_batch(D, offsets, aggregation=self.aggregation, replication=self.replication, query=False)
        if self.linear:
            #apply the linear layer (if there is one)
            D = self.linear(D)

        #is used to mask the embeddings corresponding to punctuation
        #is not used in our case
        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask

        #length normalisation
        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def doc_without_linear(self, input_ids, attention_mask):
        # used in ../evaluation/clustering.py to obtain the document embeddings before the linear layer
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]

        return D

    def score(self, Q, D):
        #implementation of the MaxSim operator
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        #masks embeddings corresponding to punctuation
        #not used in our case
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
