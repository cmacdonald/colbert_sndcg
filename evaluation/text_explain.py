import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyterrier as pt
pt.init()

from colbert.utils.parser import Arguments
from checkpoints import CHECKPOINTS
from pyterrier_colbert.ranking import ColBERTModelOnlyFactory

"""
Generate text_explains for a model thoughout training
"""

parser = Arguments(description='Re-ranking over a ColBERT index')
parser.add_model_parameters()
parser.add_argument('--model', dest='model', required=True)
parser.add_argument('--save_path', dest='save_path', default=None)
args = parser.parse()

checkpoint_0k = CHECKPOINTS[args.model+'_0k']
checkpoint_1k = CHECKPOINTS[args.model+'_1k']
checkpoint_2k = CHECKPOINTS[args.model+'_2k']
checkpoint_5k = CHECKPOINTS[args.model+'_5k']
checkpoint_10k = CHECKPOINTS[args.model+'_10k']
checkpoint_20k = CHECKPOINTS[args.model+'_20k']
checkpoint_30k = CHECKPOINTS[args.model+'_30k']
checkpoint_40k = CHECKPOINTS[args.model+'_40k']
checkpoint_50k = CHECKPOINTS[args.model+'_50k']
checkpoint_60k = CHECKPOINTS[args.model+'_60k']
checkpoint_70k = CHECKPOINTS[args.model+'_70k']
checkpoint_80k = CHECKPOINTS[args.model+'_80k']
checkpoint_90k = CHECKPOINTS[args.model+'_90k']
checkpoint_100k = CHECKPOINTS[args.model+'_100k']
checkpoint_150k = CHECKPOINTS[args.model+'_150k']
checkpoint_200k = CHECKPOINTS[args.model+'_200k']

if args.linear_layer=='none':
    dim=768
else:
    dim=128

factory_0k = ColBERTModelOnlyFactory(checkpoint_0k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_1k = ColBERTModelOnlyFactory(checkpoint_1k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_2k = ColBERTModelOnlyFactory(checkpoint_2k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_5k = ColBERTModelOnlyFactory(checkpoint_5k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_10k = ColBERTModelOnlyFactory(checkpoint_10k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_20k = ColBERTModelOnlyFactory(checkpoint_20k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_30k = ColBERTModelOnlyFactory(checkpoint_30k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_40k = ColBERTModelOnlyFactory(checkpoint_40k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_50k = ColBERTModelOnlyFactory(checkpoint_50k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_60k = ColBERTModelOnlyFactory(checkpoint_60k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_70k = ColBERTModelOnlyFactory(checkpoint_70k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_80k = ColBERTModelOnlyFactory(checkpoint_80k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_90k = ColBERTModelOnlyFactory(checkpoint_90k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_100k = ColBERTModelOnlyFactory(checkpoint_100k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_150k = ColBERTModelOnlyFactory(checkpoint_150k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)
factory_200k = ColBERTModelOnlyFactory(checkpoint_200k, linear_layer=args.linear_layer, aggregation=args.aggregation, replication=args.replication, dim=dim)


#query = "why did the us voluntarily enter ww1"
#passage = "the USA entered ww2 because of pearl harbor"
#passage = "the reason the USA entered ww2 is pearl harbor"

query = "why does my goldfish not grow"
#passage = "the growth of goldfish is limited due to tank size"
passage = "since tank size limits the growth of goldfish"

#query = "types of dysarthria from cerebral palsy"
#passage = "three major types of dysarthria in cerebral palsy: spastic, dyskinetic (athetosis) and ataxic"
#query = "do goldfish grow"
#passage = "the growth of goldfish is limited by the size of their tank"

print('After 0k')
factory_0k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_0k.png'))
print('After 1k')
factory_1k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_1k.png'))
print('After 2k')
factory_2k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_2k.png'))
print('After 5k')
factory_5k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_5k.png'))
print('After 10k')
factory_10k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_10k.png'))
print('After 20k')
factory_20k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_20k.png'))
print('After 30k')
factory_30k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_30k.png'))
print('After 40k')
factory_40k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_40k.png'))
print('After 50k')
factory_50k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_50k.png'))
print('After 60k')
factory_60k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_60k.png'))
print('After 70k')
factory_70k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_70k.png'))
print('After 80k')
factory_80k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_80k.png'))
print('After 90k')
factory_90k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_90k.png'))
print('After 100k')
factory_100k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_100k.png'))
print('After 150k')
factory_150k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_150k.png'))
print('After 200k')
factory_200k.explain_text(query, passage).savefig(os.path.join(args.save_path,args.model+'_200k.png'))