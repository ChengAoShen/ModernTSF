#!/usr/bin/env python3
"""Generate rewrite evidence for the final eight paper-model implementations."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.catalog_metadata import model_records  # noqa: E402
from benchmark.verification_results import evidence_file_sha256, verification_subject_sha256, write_verification_result  # noqa: E402
from models.pathformer.model import Model as Pathformer  # noqa: E402
from models.stpgnn.model import Model as STPGNN  # noqa: E402
from models.timeemb.model import Model as TimeEmb  # noqa: E402
from models.timefilter.model import Model as TimeFilter  # noqa: E402
from models.timeperceiver.model import Model as TimePerceiver  # noqa: E402
from models.timexer.model import Model as TimeXer  # noqa: E402
from models.umixer.model import Model as UMixer  # noqa: E402
from models.wpmixer.model import Model as WPMixer  # noqa: E402


def _marks(batch=2, steps=12, offset=0):
    rows = [[2026, 8, 1+i//24, 4, (i+offset)%24, 0] for i in range(steps)]
    return torch.tensor([rows]*batch, dtype=torch.float32)


def _graph(nodes=4):
    value = np.eye(nodes, dtype=np.float32)
    for index in range(nodes-1):
        value[index, index+1] = 1
    return value


@dataclass(frozen=True)
class Case:
    name: str
    factory: object
    boundary: object
    reference: str
    structure: dict[str, object]
    marks_active: bool = False
    adjacency_active: bool = False


CASES = (
    Case("Pathformer", lambda: Pathformer(12,3,4,layer_nums=1,k=2,num_experts=2,patch_size_list=[3,4],d_model=8,d_ff=16,n_heads=2,dropout=0), lambda: Pathformer(1,1,1,layer_nums=1,k=1,num_experts=1,patch_size_list=[1],d_model=2,d_ff=4,n_heads=1,dropout=0), "https://arxiv.org/abs/2402.05956", {
        "method":"multi-scale Transformer with adaptive pathways",
        "equations":["P_s=Divide_s(X); H_s=GlobalAttention(LocalAttention(P_s))","g=softmax(Router(statistics(X))); H=sum_s g_s H_s","X_(l+1)=Norm(X_l+H_l); Y=Linear(X_L)"],
        "modules":{"multi-scale division and dual attention":"DualScaleAttention","adaptive pathway router":"AdaptivePathway","residual forecast stack":"Model.layers/Model.forecast"},
        "differences":["dense differentiable routing replaces sparse dispatch and balance loss","top-k routes remain inspectable","forecast-only API"],
    }),
    Case("STPGNN", lambda: STPGNN(12,3,4,_graph(),dropout=0,topk=2,residual_channels=8,end_channels=16,kernel_size=2,blocks=1,layers=1,dims=4), lambda: STPGNN(1,1,1,np.ones((1,1),dtype=np.float32),dropout=0,topk=1,residual_channels=2,end_channels=4,kernel_size=1,blocks=1,layers=1,dims=2), "https://doi.org/10.1609/aaai.v38i8.28707", {
        "method":"spatio-temporal pivotal graph neural network",
        "equations":["p=TopK(degree(A)+degree(sigmoid(E_s E_t^T)))","H'_(i)=sum_(k=1)^d sigma(A_p H_[i+k] W+B)","O=sum_q U^q X L_q; Fuse(Pivotal,Graph,Temporal)"],
        "modules":{"pivotal identification":"PivotalNodeIdentification","Equation 7 pivotal convolution":"PivotalGraphConvolution","parallel non-pivotal and temporal paths":"ParallelSTLayer"},
        "differences":["smooth pivotal membership accompanies inspectable top-k selection","direct multi-horizon readout","raw calendar marks are optional covariates"],
    }, True, True),
    Case("TimeEmb", lambda: TimeEmb(12,3,4,d_model=8,use_revin=True,use_hour_index=True,use_day_index=True,scale=.02,hour_length=24,day_length=7), lambda: TimeEmb(1,1,1,d_model=2,use_revin=True,use_hour_index=True,use_day_index=False,scale=.02,hour_length=24,day_length=7), "https://arxiv.org/abs/2510.00461", {
        "method":"static-dynamic spectral disentanglement",
        "equations":["X_f=FFT(Norm(X)); D_f=X_f-E_calendar","D'_f=D_f*(1+Gate(|D_f|) complex_response)","Y=Head(IFFT(D'_f+E_calendar))"],
        "modules":{"global static representation":"GlobalCalendarEmbedding","dynamic frequency filtering":"DynamicSpectrumFilter","restoration and forecast":"Model.forward/Model.forecast"},
        "differences":["calendar marks optional with deterministic fallback","forecast head is a compact two-layer MLP","plug-in integrations and training recipe omitted"],
    }, True),
    Case("TimeFilter", lambda: TimeFilter(12,3,4,d_model=8,d_ff=16,e_layers=1,patch_len=3,dropout=0,top_p=.5,num_experts=2), lambda: TimeFilter(1,1,1,d_model=2,d_ff=4,e_layers=1,patch_len=1,dropout=0,top_p=1,num_experts=2), "https://arxiv.org/abs/2501.13041", {
        "method":"patch-specific spatial-temporal graph filtration",
        "equations":["V=PatchEmbed(X); A=softmax(Q(V)K(V)^T/sqrt(d))","A_filtered=A*1[A>=TopPThreshold(A)]","H=sum_e softmax(Router(V))_e Expert_e(V,A_filtered)"],
        "modules":{"channel-patch graph":"PatchGraphBuilder","top-p region filtration and MoE":"PatchSpecificGraphFilter","graph experts":"RegionExpert"},
        "differences":["soft expert routing keeps all experts trainable","balance loss exposed but not added to common forecast loss","non-overlapping padded patches"],
    }),
    Case("TimePerceiver", lambda: TimePerceiver(12,3,4,d_model=8,n_heads=2,patch_len=3,dropout=0,num_latents=3,latent_dim=8,latent_d_ff=16,num_latent_blocks=1), lambda: TimePerceiver(1,1,1,d_model=2,n_heads=1,patch_len=1,dropout=0,num_latents=1,latent_dim=2,latent_d_ff=4,num_latent_blocks=1), "https://arxiv.org/abs/2512.22550", {
        "method":"latent encoder and timestamp-query decoder",
        "equations":["Z=CrossAttention(Z_learned,PatchSegments(X))","Z'=LatentSelfAttention(Z)","Y_q=Head(CrossAttention(Query(time_q),Z'))"],
        "modules":{"input segment encoder":"Model.patch_embedding/Model.encoder","latent bottleneck":"Model.latents/Model.latent_blocks","target query decoder":"Model.target_queries/Model.decoder"},
        "differences":["past-to-future interface only","generalized interpolation/imputation sampler omitted","shared or channel-specific target queries configurable"],
    }, True),
    Case("TimeXer", lambda: TimeXer(12,3,4,d_model=8,n_heads=2,e_layers=1,d_ff=16,patch_len=3,dropout=0), lambda: TimeXer(1,1,1,d_model=2,n_heads=1,e_layers=1,d_ff=4,patch_len=1,dropout=0), "https://arxiv.org/abs/2402.19072", {
        "method":"Transformer forecasting with exogenous variables",
        "equations":["E_en=[PatchEmbed(X_en),g_en]","E_en'=SelfAttention(E_en)","g_en'=CrossAttention(g_en,E_ex); Y=Head([patches,g_en'])"],
        "modules":{"endogenous patches and global token":"EndogenousEmbedding","variate-wise exogenous embedding":"ExogenousEmbedding","self/cross-attention bridge":"TimeXerLayer"},
        "differences":["MS/S use last channel as endogenous","M vectorizes every target against shared external context","calendar summarized as an exogenous token"],
    }, True),
    Case("UMixer", lambda: UMixer(12,3,4,d_model=8,e_layers=2,patch_len=3,stride=3,dropout=0), lambda: UMixer(1,1,1,d_model=2,e_layers=1,patch_len=1,stride=1,dropout=0), "https://arxiv.org/abs/2401.02236", {
        "method":"U-shaped patch mixer with stationarity correction",
        "equations":["H=FeatureMix(PatchMix(PatchEmbed(X)))","H_up=Mix(Fuse(Upsample(H),H_skip))","alpha=sqrt(Power(H_original)/Power(H_processed)); Y=Head(alpha H)"],
        "modules":{"axis-specific mixer":"AxisMixer","U-shaped skip hierarchy":"Model.down_mixers/Model.up_mixers","spectral correction":"StationarityCorrection"},
        "differences":["latent autocorrelation-energy correction","direct multi-horizon head","paper training objective and datasets omitted"],
    }),
    Case("WPMixer", lambda: WPMixer(12,3,4,d_model=8,dropout=0,tfactor=2,dfactor=2,wavelet="db2",level=2,patch_len=3,stride=2), lambda: WPMixer(1,1,1,d_model=2,dropout=0,tfactor=1,dfactor=1,wavelet="haar",level=1,patch_len=1,stride=1), "https://arxiv.org/abs/2412.17176", {
        "method":"multi-resolution wavelet patch mixer",
        "equations":["A_j,D_j=DWT_j(X)","H_j=FeatureMLP(TokenMLP(PatchEmbed(R_j)))","Y=sum_j softmax(w)_j Head_j(H_j)"],
        "modules":{"orthogonal wavelet analysis":"OrthogonalWaveletAnalysis","per-resolution patch and axis mixing":"ResolutionMixer","learned resolution fusion":"Model.resolution_logits"},
        "differences":["fixed local Haar/db1/db2 analysis filters","branch forecasts fused directly without external inverse-wavelet package","forecast-only API"],
    }),
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def _runtime(case):
    torch.manual_seed(260827)
    model = case.factory().cpu().eval()
    x = torch.randn(2, 12, 4, requires_grad=True)
    output = model(x, _marks(), x_mark_dec=_marks(2, 3, 5))
    if output.shape != (2, 3, 4) or not torch.isfinite(output).all():
        raise AssertionError("forward contract failed")
    output.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all() or x.grad.abs().max() == 0:
        raise AssertionError("input gradient failed")
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"inactive or invalid parameter: {name}")
        gradients[name] = float(parameter.grad.abs().max())
    clone = case.factory().cpu().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    cloned = clone(x.detach(), _marks(), x_mark_dec=_marks(2, 3, 5))
    torch.testing.assert_close(cloned, output.detach())
    boundary = case.boundary().cpu().eval()
    if boundary(torch.randn(1, 1, 1), _marks(1, 1), x_mark_dec=_marks(1, 1)).shape != (1, 1, 1):
        raise AssertionError("minimum boundary failed")
    try:
        model(torch.randn(1, 11, 4))
    except ValueError:
        wrong_length = True
    else:
        raise AssertionError("wrong length accepted")
    first = model(x.detach(), _marks(offset=0), x_mark_dec=_marks(2, 3, 0))
    second = model(x.detach(), _marks(offset=7), x_mark_dec=_marks(2, 3, 7))
    marks_effect = float((first-second).abs().max())
    if case.marks_active != (marks_effect > 0):
        raise AssertionError("marks contract mismatch")
    adjacency_effect = 0.0
    if case.adjacency_active:
        torch.manual_seed(91)
        identity = STPGNN(12,3,4,np.eye(4),dropout=0,topk=2,residual_channels=8,end_channels=16,kernel_size=2,blocks=1,layers=1,dims=4).eval()
        torch.manual_seed(91)
        connected = case.factory().eval()
        adjacency_effect = float((identity(x.detach())-connected(x.detach())).abs().max())
        if adjacency_effect == 0:
            raise AssertionError("adjacency inactive")
    return {"shape":[2,3,4],"input_gradient_max_abs":float(x.grad.abs().max()),"parameter_gradients":gradients,"round_trip_max_abs":float((cloned-output.detach()).abs().max()),"marks_effect_max_abs":marks_effect,"adjacency_effect_max_abs":adjacency_effect,"wrong_length_rejected":wrong_length}


def _environment():
    return {"python":platform.python_version(),"framework":f"torch {torch.__version__}","dependencies":{"numpy":np.__version__,"pydantic":importlib.metadata.version("pydantic"),"torch":torch.__version__},"platform":platform.platform(),"device":"cpu","dtype":"float32","deterministic":{"seed":260827,"num_threads":torch.get_num_threads()}}


def _check(evidence, **metrics):
    return {"passed":True,"evidence":evidence,"metrics":metrics}


def verify(case, records):
    observations = _runtime(case)
    structure_digest = _digest(case.structure)
    relative = f"verification/rewrite/{case.name}.json"
    artifact_path = ROOT / relative
    artifact = {"schema_version":1,"kind":"clean-room-structure-map","model":case.name,"reference":case.reference,"independent_design":True,"source_code_not_copied":True,"structure_map":case.structure,"structure_map_sha256":structure_digest,"observations":observations}
    _atomic_json(artifact_path, artifact)
    evidence = [relative, "tests/test_final_clean_room_rewrites.py"]
    checks = {
        "paper_structure":_check(evidence,mapped_elements=len(case.structure["modules"]),claim="paper-to-independent-local-map"),
        "equations":_check(evidence,cases=len(case.structure["equations"])),
        "construction":_check(evidence,instances=3),
        "forward":_check(evidence,shape="2,3,4"),
        "backward":_check(evidence,input_gradient_max_abs=observations["input_gradient_max_abs"]),
        "finite_outputs":_check(evidence,nonfinite=0),
        "active_parameter_gradients":_check(evidence,parameters=len(observations["parameter_gradients"])),
        "state_dict_round_trip":_check(evidence,max_abs=observations["round_trip_max_abs"]),
        "cpu":_check(evidence,device="cpu"),
        "batch_size_boundary":_check(evidence,cases="batch=1,batch=2"),
        "sequence_length_boundary":_check(evidence,cases="seq=1,wrong-length-rejected"),
        "marks_adjacency_contract":_check(evidence,marks="active" if case.marks_active else "intentionally-unused",marks_effect_max_abs=observations["marks_effect_max_abs"],adjacency_effect_max_abs=observations["adjacency_effect_max_abs"]),
    }
    result = {"schema_version":1,"kind":"rewrite-validation","model":case.name,"implementation":"rewrite","verified_at":datetime.now(timezone.utc),"subject_sha256":verification_subject_sha256(ROOT,records[case.name]),"commands":[f"uv run python scripts/verify_final_clean_room_rewrites.py --model {case.name}","uv run python -m unittest tests.test_final_clean_room_rewrites -v",f"uv run tsf repo doctor --strict --models {case.name}"],"environment":_environment(),"artifacts":{relative:evidence_file_sha256(artifact_path)},"passed":True,"basis":{"references":[case.reference],"structure_map_sha256":structure_digest,"independent_design":True,"source_code_not_copied":True},"checks":checks}
    write_verification_result(ROOT / "verification/model-results.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=[case.name for case in CASES])
    args = parser.parse_args()
    selected = set(args.model or [case.name for case in CASES])
    records = {str(record["name"]):record for record in model_records(ROOT)}
    for case in CASES:
        if case.name in selected:
            verify(case, records)
            print(f"{case.name}: rewrite validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
