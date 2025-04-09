#Download URLs of ColBERT checkpoints
from huggingface_hub import hf_hub_url

CHECKPOINTS = {}

#linear_layer='none'
CHECKPOINTS['ll_none_0k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-1.dnn')
CHECKPOINTS['ll_none_1k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-1000.dnn')
CHECKPOINTS['ll_none_2k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-2000.dnn')
CHECKPOINTS['ll_none_5k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-5000.dnn')
CHECKPOINTS['ll_none_10k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-10000.dnn')
CHECKPOINTS['ll_none_20k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-20000.dnn')
CHECKPOINTS['ll_none_25k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-25000.dnn')
CHECKPOINTS['ll_none_30k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-30000.dnn')
CHECKPOINTS['ll_none_40k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-40000.dnn')
CHECKPOINTS['ll_none_50k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-50000.dnn')
CHECKPOINTS['ll_none_60k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-60000.dnn')
CHECKPOINTS['ll_none_70k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-70000.dnn')
CHECKPOINTS['ll_none_75k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-75000.dnn')
CHECKPOINTS['ll_none_80k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-80000.dnn')
CHECKPOINTS['ll_none_90k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-9000.dnn')
CHECKPOINTS['ll_none_100k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-100000.dnn')
CHECKPOINTS['ll_none_150k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-150000.dnn')
CHECKPOINTS['ll_none_200k'] = hf_hub_url('ArianeS21/colbert_sndcg','colbert-200000.dnn')
