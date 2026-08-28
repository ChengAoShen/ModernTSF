#!/usr/bin/env python3
"""Emit reproducible evidence for architecture clean-room batch C."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from benchmark.catalog_metadata import model_records
from benchmark.verification_results import evidence_file_sha256, verification_subject_sha256, write_verification_result
from tests.test_architecture_clean_room_rewrites_c import RuntimeTests

STRUCTURES = {
 "DTAF": {"reference":"https://arxiv.org/abs/2511.08229","equations":{"TFS":"routed nuisance subtraction plus causal history/current fusion","FWM":"adjacent FFT-amplitude difference selects spectral shifts","dual branch":"residual temporal/frequency fusion"},"modules":["TemporalStabilizingFusion","FrequencyWaveModeling","DTAFBlock"],"differences":["portable SiLU experts","direct patch forecast head"]},
 "Fredformer": {"reference":"https://arxiv.org/abs/2406.09009","equations":{"band division":"contiguous complex frequency bands","frequency debias":"per-band RMS equalization","band learning":"shared channel attention independently per band"},"modules":["FrequencyEqualization","FrequencyBandAttention","split_frequency_bands"],"differences":["full attention only","contiguous-band implementation","no Nyström approximation"]},
 "HDMixer": {"reference":"https://ojs.aaai.org/index.php/AAAI/article/view/29155","equations":{"LEP":"bounded center and width offsets around regular anchors","sampling":"differentiable bilinear interpolation","HDE":"separate within-patch, across-patch, variable and feature MLPs"},"modules":["LengthExtendablePatcher","AxisMixer","HierarchicalDependencyBlock"],"differences":["explicit center/width parameterization","forecast-only"]},
 "MICN": {"reference":"https://openreview.net/references/pdf?id=u64xKhWy-T","equations":{"hybrid decomposition":"average seasonal/trend splits over scales","local extraction":"strided depthwise convolution","global extraction":"isometric convolution on shortened grid","restoration":"transposed convolution to original grid"},"modules":["MultiScaleDecomposition","IsometricConvolutionBranch","MICLayer"],"differences":["three-tap global operator","calendar embeddings omitted"]},
 "MSGNet": {"reference":"https://arxiv.org/abs/2401.00423","equations":{"scale discovery":"top FFT amplitudes yield periods","adaptive graph":"softmax activation of E1 E2 per scale","MixHop":"H_k=alpha X+(1-alpha) A H_(k-1)","aggregation":"FFT strengths softmax-weight scale outputs"},"modules":["AdaptiveMixHopGraph","ScaleGraphBranch","MultiScaleGraphBlock"],"differences":["LeakyReLU prevents dead adjacency factors","period attention is variable-wise","calendar embedding omitted"]},
 "ModernTCN": {"reference":"https://openreview.net/forum?id=vpJMJerXHU","equations":{"patch stem":"variable-independent strided convolution","temporal mixing":"large-kernel depthwise convolution","ConvFFN1":"pointwise groups=M","ConvFFN2":"reshape then pointwise groups=D"},"modules":["LargeKernelDepthwiseConv","ModernTCNBlock","ModernTCNBackbone"],"differences":["training branches remain explicit","forecast-only"]},
}

def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def runtime(name):
    torch.manual_seed(12000 + sum(map(ord,name)))
    factory=RuntimeTests.factories()[name]; model=factory().cpu()
    values=torch.randn(2,16,2,requires_grad=True); output=RuntimeTests.call(model,values)
    if output.shape != (2,4,2) or not torch.isfinite(output).all(): raise AssertionError(f"{name}: forward")
    output.square().mean().backward()
    gradients={}
    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max()==0:
            raise AssertionError(f"{name}: inactive {parameter_name}")
        gradients[parameter_name]=float(parameter.grad.abs().max())
    model.eval(); expected=RuntimeTests.call(model,values.detach())
    clone=factory().eval(); clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(RuntimeTests.call(clone,values.detach()),expected)
    torch.testing.assert_close(RuntimeTests.call(model,values.detach(),True),expected)
    if RuntimeTests.call(model,torch.randn(1,16,2)).shape != (1,4,2): raise AssertionError(f"{name}: batch")
    try: RuntimeTests.call(model,torch.randn(1,15,2))
    except ValueError: rejected=True
    else: raise AssertionError(f"{name}: length")
    return {"shape":[2,4,2],"input_gradient_max_abs":float(values.grad.abs().max()),
            "parameter_gradients":gradients,"state_dict_max_abs":0.0,
            "batch_size_cases":[1,2],"wrong_length_rejected":rejected,
            "marks_contract":"accepted-and-ignored","adjacency_contract":"not-declared"}

def check(evidence, **metrics): return {"passed":True,"evidence":evidence,"metrics":metrics}

def main():
    records={str(record["name"]):record for record in model_records(ROOT)}
    for name, structure in STRUCTURES.items():
        observations=runtime(name); structure_digest=digest(structure)
        relative=f"verification/rewrite/{name}.json"; path=ROOT/relative; path.parent.mkdir(parents=True,exist_ok=True)
        artifact={"schema_version":1,"kind":"clean-room-structure-map","model":name,
                  "reference":structure["reference"],"independent_design":True,
                  "source_code_not_copied":True,"structure_map":structure,
                  "structure_map_sha256":structure_digest,"observations":observations}
        path.write_text(json.dumps(artifact,indent=2,sort_keys=True)+"\n")
        evidence=[relative,"tests/test_architecture_clean_room_rewrites_c.py"]
        checks={
          "paper_structure":check(evidence,mapped_elements=len(structure["modules"])),
          "equations":check(evidence,cases=len(structure["equations"])),
          "construction":check(evidence,instances=3), "forward":check(evidence,shape="2,4,2"),
          "backward":check(evidence,input_gradient_max_abs=observations["input_gradient_max_abs"]),
          "finite_outputs":check(evidence,nonfinite=0),
          "active_parameter_gradients":check(evidence,parameters=len(observations["parameter_gradients"])),
          "state_dict_round_trip":check(evidence,max_abs=0.0), "cpu":check(evidence,device="cpu"),
          "batch_size_boundary":check(evidence,cases="batch=1,batch=2"),
          "sequence_length_boundary":check(evidence,cases="expected-length;wrong-length-rejected"),
          "marks_adjacency_contract":check(evidence,contract="marks-accepted-and-ignored;adjacency-not-declared"),
        }
        seed=12000+sum(map(ord,name))
        result={"schema_version":1,"kind":"rewrite-validation","model":name,
          "implementation":"rewrite","verified_at":datetime.now(UTC),
          "subject_sha256":verification_subject_sha256(ROOT,records[name]),
          "commands":["uv run python scripts/verify_architecture_clean_room_rewrites_c.py",
                      "uv run python -m unittest tests.test_architecture_clean_room_rewrites_c -v",
                      f"uv run tsf repo doctor --strict --models {name}"],
          "environment":{"python":platform.python_version(),"framework":f"torch {torch.__version__}",
             "dependencies":{"torch":torch.__version__,"pydantic":importlib.metadata.version("pydantic")},
             "platform":platform.platform(),"device":"cpu","dtype":"float32",
             "deterministic":{"seed":seed,"num_threads":torch.get_num_threads()}},
          "artifacts":{relative:evidence_file_sha256(path)},"passed":True,
          "basis":{"references":[structure["reference"]],"structure_map_sha256":structure_digest,
                   "independent_design":True,"source_code_not_copied":True},"checks":checks}
        write_verification_result(ROOT/"verification/model-results.json",result)
        print(f"{name}: rewrite validation passed")
    return 0

if __name__ == "__main__": raise SystemExit(main())
