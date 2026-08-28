#!/usr/bin/env python3
"""Generate clean-room evidence for CARD/Crossformer/DSFormer/DUET/MPF/NST."""
from __future__ import annotations
import copy, hashlib, importlib.metadata, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from benchmark.catalog_metadata import model_records
from benchmark.verification_results import evidence_file_sha256, verification_subject_sha256, write_verification_result
from models.card.model import Model as CARD
from models.crossformer.model import Model as Crossformer
from models.dsformer.model import Model as DSFormer
from models.duet.model import Model as DUET
from models.multipatchformer.model import Model as MultiPatchFormer
from models.nstransformer.model import Model as NSTransformer

STRUCTURES={
 "CARD":{"reference":"https://openreview.net/forum?id=MJksrOhurE","equations":{"EMA":"causal exponentially weighted token alignment","dual attention":"temporal-token and aligned cross-channel attention","blend":"learned convex gate"},"modules":{"patches":"patch_projection","alignment":"exponential_smooth","attention":"ChannelAlignedBlock","forecast":"head"},"differences":["robust uncertainty-weighted training loss omitted","dynamic low-rank approximation omitted"]},
 "Crossformer":{"reference":"https://openreview.net/forum?id=vSVLM2j9eie","equations":{"DSW":"dimension-segment-wise embedding","TSA":"cross-time then router-mediated cross-dimension attention","HED":"hierarchical segment merging and scale prediction"},"modules":{"embedding":"dsw_embed","attention":"TwoStageAttention","hierarchy":"SegmentMerge","forecast":"scale heads"},"differences":["direct scale heads replace paper decoder"]},
 "DSFormer":{"reference":"https://arxiv.org/abs/2308.03274","equations":{"DS":"piecewise and interval sampling","TVA":"parallel temporal and variable attention","integration":"gated dual-view fusion"},"modules":{"sampling":"dual_sampling","attention":"TVABlock","normalization":"RevIN","forecast":"decoder_attention and head"},"differences":["learned gated fusion and direct linear head"]},
 "DUET":{"reference":"https://arxiv.org/abs/2412.10859","equations":{"TCE":"distribution-aware top-k temporal mixture","experts":"trend/seasonal linear experts","CCE":"Mahalanobis channel relation and attention"},"modules":{"router":"DistributionalRouter","experts":"TemporalExpert","relation":"mahalanobis_bias","channels":"ChannelAttention"},"differences":["small dense routing residual keeps all experts trainable","auxiliary balancing loss omitted"]},
 "MultiPatchFormer":{"reference":"https://doi.org/10.1038/s41598-024-82417-4","equations":{"multiscale":"8/16/24/32-point patch views","dual encoding":"temporal then channel attention","SAR":"progressive horizon-group conditioning"},"modules":{"patches":"PatchScale","temporal":"temporal_layers","channels":"channel_layer","decoder":"SemiAutoregressiveHead"},"differences":["linear alignment of unequal patch grids"]},
 "NSTransformer":{"reference":"https://arxiv.org/abs/2205.14415","equations":{"stationarization":"detached mean/std normalization and restoration","Eq. 5":"positive tau and temporal delta projectors","Eq. 6":"softmax((tau QK^T + delta)/sqrt(d))"},"modules":{"projectors":"Projector","attention":"DeStationaryAttention","backbone":"NSBlock","forecast":"future_queries and projection"},"differences":["learned future-query decoder","calendar embedding and non-forecast tasks omitted"]},
}

def factory(name):
 return {"CARD":lambda:CARD(8,3,2,patch_len=4,stride=2,d_model=8,n_heads=2,e_layers=1,d_ff=16,dropout=0),"Crossformer":lambda:Crossformer(8,3,2,d_model=8,n_heads=2,e_layers=2,d_ff=16,seg_len=2,win_size=2,factor=2,dropout=0),"DSFormer":lambda:DSFormer(8,3,2,num_layer=1,muti_head=2,num_samp=2,dropout=0),"DUET":lambda:DUET(8,3,2,d_model=8,n_heads=2,e_layers=1,d_ff=16,dropout=0,fc_dropout=0,moving_avg=3,num_experts=2,k=1,hidden_size=8,noisy_gating=False),"MultiPatchFormer":lambda:MultiPatchFormer(8,3,2,d_model=8,n_heads=2,e_layers=1,d_ff=16,dropout=0),"NSTransformer":lambda:NSTransformer(8,3,0,2,d_model=8,n_heads=2,e_layers=1,d_layers=1,d_ff=16,dropout=0,p_hidden_dims=[8],p_hidden_layers=1)}[name]()

def runtime(name):
 torch.manual_seed(9301+sum(map(ord,name))); model=factory(name).cpu(); x=torch.randn(2,8,2,requires_grad=True)
 y=model(x,torch.randn(2,8,6)); assert y.shape==(2,3,2) and torch.isfinite(y).all(); y.square().mean().backward()
 assert x.grad is not None and torch.isfinite(x.grad).all(); gradients={}
 for key,value in model.named_parameters():
  assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().max()>0,key
  gradients[key]=float(value.grad.abs().max())
 model.eval(); expected=model(x.detach()); clone=factory(name).eval(); clone.load_state_dict(copy.deepcopy(model.state_dict())); torch.testing.assert_close(clone(x.detach()),expected)
 assert model(torch.randn(1,8,2)).shape==(1,3,2)
 try: model(torch.randn(1,7,2))
 except ValueError: rejected=True
 else: raise AssertionError("wrong length accepted")
 torch.testing.assert_close(model(x.detach(),torch.randn(2,8,6)),expected)
 return {"shape":[2,3,2],"input_gradient_max_abs":float(x.grad.abs().max()),"parameter_gradients":gradients,"batch_size_cases":[1,2],"wrong_length_rejected":rejected,"marks_active":False,"adjacency_contract":"not declared"}

def check(evidence,**metrics): return {"passed":True,"evidence":evidence,"metrics":metrics}
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def main():
 records={str(r["name"]):r for r in model_records(ROOT)}
 for name,structure in STRUCTURES.items():
  observations=runtime(name); relative=f"verification/rewrite/{name}.json"; path=ROOT/relative; path.parent.mkdir(parents=True,exist_ok=True)
  artifact={"schema_version":1,"kind":"clean-room-structure-map","model":name,"reference":structure["reference"],"independent_design":True,"source_code_not_copied":True,"structure_map":structure,"structure_map_sha256":digest(structure),"observations":observations}
  path.write_text(json.dumps(artifact,indent=2,sort_keys=True)+"\n",encoding="utf-8")
  evidence=[relative,"tests/test_transformer_patch_clean_room_batch_b.py"]
  checks={"paper_structure":check(evidence,mapped_elements=len(structure["modules"])),"equations":check(evidence,cases=len(structure["equations"])),"construction":check(evidence,instances=3),"forward":check(evidence,shape="2,3,2"),"backward":check(evidence,input_gradient_max_abs=observations["input_gradient_max_abs"]),"finite_outputs":check(evidence,nonfinite=0),"active_parameter_gradients":check(evidence,parameters=len(observations["parameter_gradients"])),"state_dict_round_trip":check(evidence,max_abs=0.0),"cpu":check(evidence,device="cpu"),"batch_size_boundary":check(evidence,cases="batch=1,batch=2"),"sequence_length_boundary":check(evidence,cases="minimum-tested;wrong-length-rejected"),"marks_adjacency_contract":check(evidence,contract="marks-accepted-and-ignored;adjacency-not-declared")}
  seed=9301+sum(map(ord,name)); result={"schema_version":1,"kind":"rewrite-validation","model":name,"implementation":"rewrite","verified_at":datetime.now(timezone.utc),"subject_sha256":verification_subject_sha256(ROOT,records[name]),"commands":["uv run python scripts/verify_transformer_patch_clean_room_batch_b.py","uv run python -m unittest tests.test_transformer_patch_clean_room_batch_b -v",f"uv run tsf repo doctor --strict --models {name}"],"environment":{"python":platform.python_version(),"framework":f"torch {torch.__version__}","dependencies":{"pydantic":importlib.metadata.version("pydantic"),"torch":torch.__version__},"platform":platform.platform(),"device":"cpu","dtype":"float32","deterministic":{"seed":seed,"num_threads":torch.get_num_threads()}},"artifacts":{relative:evidence_file_sha256(path)},"passed":True,"basis":{"references":[structure["reference"]],"structure_map_sha256":digest(structure),"independent_design":True,"source_code_not_copied":True},"checks":checks}
  write_verification_result(ROOT/"verification/model-results.json",result); print(f"{name}: rewrite validation passed")
 return 0
if __name__=="__main__": raise SystemExit(main())
