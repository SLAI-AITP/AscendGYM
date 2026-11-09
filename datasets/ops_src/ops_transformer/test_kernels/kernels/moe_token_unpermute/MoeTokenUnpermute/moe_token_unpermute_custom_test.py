import os
import torch
import torch_npu
import numpy as np
import gtsrusops as ops

DEVICE = "npu:0"
GOLDEN_INPUT_PATH = os.environ.get('ASCENDGYM_GOLDEN_INPUT_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data/golden_input/MoeTokenUnpermute'))
GOLDEN_RESULT_PATH = os.environ.get('ASCENDGYM_GOLDEN_RESULT_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data/golden_result/MoeTokenUnpermute'))

# ── ground_truth reference implementation (for reference only; comparison baseline is golden_result files)─────────

import torch

def moe_token_unpermute(permuted_tokens, sorted_indices, probs=None):
    sorted_indices = sorted_indices.long()
    if probs is not None:
        topK = probs.shape[1]
        tokens_num = probs.shape[0]
    else:
        topK = 1
        tokens_num = sorted_indices.shape[0]
    T = permuted_tokens[sorted_indices]
    if probs is not None:
        probs_expanded = probs.view(-1, 1).expand_as(T)
        T = T * probs_expanded
        O = T.view(tokens_num, topK, -1).sum(dim=1)
    else:
        O = T
    return O



PARAM_NAMES = ['permuted_tokens', 'sorted_indices', 'probs', 'padded_mode']
FORWARD_DEFAULTS = {}
OUTPUT_KEYS = ['unpermuted_tokens']

# ── Case Specs ─────────────────────────────────────────────────────────────
# case_0: in=[permuted_tokens:float16[16, 64], sorted_indices:int32[16], probs:float16[8, 2], padded_mode=False] -> out=[unpermuted_tokens:float16]
# case_1: in=[permuted_tokens:float16[64, 128], sorted_indices:int32[64], probs:float16[16, 4], padded_mode=False] -> out=[unpermuted_tokens:float16]
# case_2: in=[permuted_tokens:bfloat16[64, 64], sorted_indices:int32[64], probs:bfloat16[32, 2], padded_mode=False] -> out=[unpermuted_tokens:bfloat16]
# case_3: in=[permuted_tokens:float32[12, 256], sorted_indices:int32[12], probs:float32[4, 3], padded_mode=False] -> out=[unpermuted_tokens:float32]
# case_4: in=[permuted_tokens:float16[8, 128], sorted_indices:int32[8], padded_mode=False] -> out=[unpermuted_tokens:float16]
# case_5: in=[permuted_tokens:bfloat16[160, 256], sorted_indices:int32[160], probs:bfloat16[20, 8], padded_mode=False] -> out=[unpermuted_tokens:bfloat16]
# case_6: in=[permuted_tokens:float16[128, 64], sorted_indices:int32[128], probs:float16[64, 2], padded_mode=False] -> out=[unpermuted_tokens:float16]
# case_7: in=[permuted_tokens:float32[512, 512], sorted_indices:int32[512], probs:float32[128, 4], padded_mode=False] -> out=[unpermuted_tokens:float32]
# case_8: in=[permuted_tokens:bfloat16[512, 128], sorted_indices:int32[512], probs:bfloat16[256, 2], padded_mode=False] -> out=[unpermuted_tokens:bfloat16]
# case_9: in=[permuted_tokens:float16[30, 32], sorted_indices:int32[30], probs:float16[10, 3], padded_mode=False] -> out=[unpermuted_tokens:float16]

case_files = sorted([
    f for f in os.listdir(GOLDEN_INPUT_PATH)
    if f.startswith("case_") and f.endswith(".pt")
], key=lambda x: int(x.replace("case_", "").replace(".pt", "")))

_ATOL_MAP = {torch.float16: 2**-8, torch.bfloat16: 2**-7, torch.float32: 2**-11}


def compare_tensor(ref_t, cus_t, name=''):
    ref_np = ref_t.float().cpu().detach().numpy().flatten().astype('float64')
    cus_np = cus_t.float().cpu().detach().numpy().flatten().astype('float64')
    diff = np.abs(ref_np - cus_np)
    rel_mask = diff / np.maximum(np.abs(ref_np), 1.0)
    atol = _ATOL_MAP.get(ref_t.dtype, 0)
    error_rate = float((rel_mask > atol).sum()) / max(len(ref_np), 1)
    max_diff = float(diff.max()) if diff.size > 0 else 0.0
    status = "PASS" if error_rate <= atol else "FAIL"
    return max_diff, status


def get_call_args(input_data, case_file):
    # Organize parameters in PARAM_NAMES order, fill missing ones with defaults
    call_args = []
    for pname in PARAM_NAMES:
        if pname in input_data:
            val = input_data[pname]
            if isinstance(val, torch.Tensor):
                call_args.append(val.to(DEVICE))
            elif isinstance(val, list):
                call_args.append([v.to(DEVICE) if isinstance(v, torch.Tensor) else v for v in val])
            else:
                call_args.append(val)
        elif pname in FORWARD_DEFAULTS:
            call_args.append(FORWARD_DEFAULTS[pname])
        else:
            raise ValueError(f"Missing required param '{pname}' in {case_file} and no default")
    return call_args


print(f"case                                               output               max_diff         status  ")
print("-" * 98)

all_pass = True

for case_file in case_files:
    case_name = case_file.replace(".pt", "")
    input_data = torch.load(os.path.join(GOLDEN_INPUT_PATH, case_file), weights_only=False)
    golden_result = torch.load(os.path.join(GOLDEN_RESULT_PATH, case_file), weights_only=False)

    call_args = get_call_args(input_data, case_file)

    # Build out_buf: in OUTPUT_KEYS order, get dtype/shape from golden_result
    out_bufs = []
    for okey in OUTPUT_KEYS:
        if okey in golden_result:
            ref = golden_result[okey]
            out_bufs.append(torch.empty_like(ref).to(DEVICE))

    if len(out_bufs) == 1:
        out_buf_arg = out_bufs[0]
    else:
        out_buf_arg = out_bufs

    out_custom = ops.npu_moe_token_unpermute_custom(*call_args, out_buf_arg)

    # Compare precision output by output
    if isinstance(out_custom, torch.Tensor):
        out_custom = [out_custom]
    elif isinstance(out_custom, tuple):
        out_custom = list(out_custom)

    for i, okey in enumerate(OUTPUT_KEYS):
        if okey not in golden_result:
            continue
        ref_t = golden_result[okey].to(DEVICE)
        cus_t = out_custom[i] if i < len(out_custom) else None
        if cus_t is None:
            na = 'N/A'; sk = 'SKIP'
            print(f"{case_name:<50} {okey:<20} {na:<16} {sk:<8}")
            continue
        max_diff, status = compare_tensor(ref_t, cus_t, name=okey)
        if status == "FAIL":
            all_pass = False
        print(f"{case_name:<50} {okey:<20} {max_diff:<16.8f} {status:<8}")

print("-" * 98)
if all_pass:
    print("All tests PASSED!")
else:
    print("Some tests FAILED!")
