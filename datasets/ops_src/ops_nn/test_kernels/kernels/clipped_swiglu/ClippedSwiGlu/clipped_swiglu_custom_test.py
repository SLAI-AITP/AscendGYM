import os
import torch
import torch_npu
import numpy as np
import gtsrusops as ops

DEVICE = "npu:0"
GOLDEN_INPUT_PATH = os.environ.get('ASCENDGYM_GOLDEN_INPUT_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data/golden_input/ClippedSwiGlu'))
GOLDEN_RESULT_PATH = os.environ.get('ASCENDGYM_GOLDEN_RESULT_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data/golden_result/ClippedSwiGlu'))

# ── ground_truth reference implementation (for reference only; comparison baseline is golden_result files)─────────
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.limit = limit
        self.bias = bias
        self.interleaved = interleaved

    def forward(self, x) -> torch.Tensor:
        if self.interleaved:
            # Interleaved: odd/even elements
            slices = [slice(None)] * x.dim()
            slices_a = slices.copy()
            slices_b = slices.copy()
            slices_a[self.dim] = slice(0, None, 2)
            slices_b[self.dim] = slice(1, None, 2)
            A = x[tuple(slices_a)]
            B = x[tuple(slices_b)]
        else:
            # Front-back: first half / second half
            split_size = x.shape[self.dim] // 2
            chunks = torch.chunk(x, 2, dim=self.dim)
            A = chunks[0]
            B = chunks[1]

        A = torch.clamp(A, min=-self.limit, max=self.limit)
        B = torch.clamp(B, min=-self.limit, max=self.limit) + self.bias
        A_sig = A / (1.0 + torch.exp(-self.alpha * A))
        y = A_sig * B
        return y



PARAM_NAMES = ['x', 'dim', 'alpha', 'limit', 'bias', 'interleaved']
FORWARD_DEFAULTS = {}
OUTPUT_KEYS = ['y']

# ── Case Specs ─────────────────────────────────────────────────────────────
# case_0: in=[x:float16[2, 32], dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True] -> out=[y:float16]
# case_1: in=[x:float16[1, 512], dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True] -> out=[y:float16]
# case_2: in=[x:float16[4, 64, 32], dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=False] -> out=[y:float16]
# case_3: in=[x:float16[8, 128, 64], dim=1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True] -> out=[y:float16]
# case_4: in=[x:float16[2, 256, 128], dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True] -> out=[y:float16]
# case_5: in=[x:float32[2, 16, 32, 64], dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True] -> out=[y:float32]
# case_6: in=[x:float32[64, 256], dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True] -> out=[y:float32]
# case_7: in=[x:float32[1, 2, 4, 128, 64], dim=2, alpha=1.702, limit=7.0, bias=1.0, interleaved=False] -> out=[y:float32]
# case_8: in=[x:bfloat16[32, 1024], dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True] -> out=[y:bfloat16]
# case_9: in=[x:bfloat16[2, 64, 128], dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True] -> out=[y:bfloat16]

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

    out_custom = ops.npu_clipped_swi_glu_custom(*call_args, out_buf_arg)

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
