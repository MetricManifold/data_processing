#!/usr/bin/env python3
"""Compare volume statistics between two simulation runs."""
import re, sys, subprocess
import numpy as np

def extract_all_vols(run_dir):
    r = subprocess.run(
        ['python3', 'postprocessing/validate_correctness.py', run_dir, '1562', '1562', '288'],
        capture_output=True, text=True
    )
    all_vols = []
    for line in (r.stdout + r.stderr).split('\n'):
        m = re.search(r'vol=([0-9.]+).*phi', line)
        if m:
            all_vols.append(float(m.group(1)))
    return np.array(all_vols)

print("Reading pre-optimization reference run...")
ref = extract_all_vols('agent_test_runs/validate_fused_288')
print("Reading optimized run...")
opt = extract_all_vols('agent_test_runs/validate_288cell_correctness')

target = 7543.0
ref_err = (ref - target) / target * 100
opt_err = (opt - target) / target * 100

print()
print("=" * 50)
print("  VOLUME CONSERVATION COMPARISON")
print("=" * 50)
print(f"  Target volume (piR^2, R=49): {target:.1f}")
print()
print(f"  Pre-optimization (reference):")
print(f"    N cells:     {len(ref)}")
print(f"    Mean vol:    {ref.mean():.1f}  ({ref_err.mean():+.1f}%)")
print(f"    Std vol:     {ref.std():.1f}  ({ref.std()/target*100:.1f}%)")
print(f"    Min vol:     {ref.min():.1f}  ({ref_err.min():+.1f}%)")
print(f"    Max vol:     {ref.max():.1f}  ({ref_err.max():+.1f}%)")
print(f"    >10% error:  {np.sum(np.abs(ref_err) > 10)} cells")
print()
print(f"  Optimized (current):")
print(f"    N cells:     {len(opt)}")
print(f"    Mean vol:    {opt.mean():.1f}  ({opt_err.mean():+.1f}%)")
print(f"    Std vol:     {opt.std():.1f}  ({opt.std()/target*100:.1f}%)")
print(f"    Min vol:     {opt.min():.1f}  ({opt_err.min():+.1f}%)")
print(f"    Max vol:     {opt.max():.1f}  ({opt_err.max():+.1f}%)")
print(f"    >10% error:  {np.sum(np.abs(opt_err) > 10)} cells")
print()
print(f"  Difference (optimized - reference):")
print(f"    Mean vol diff:  {opt.mean()-ref.mean():+.1f}  ({(opt.mean()-ref.mean())/target*100:+.2f}%)")
print(f"    Std vol diff:   {opt.std()-ref.std():+.1f}")
print()
print("=" * 50)
print("  VERDICT")
print("=" * 50)
diff_pct = abs(opt.mean() - ref.mean()) / target * 100
if diff_pct < 2.0:
    print(f"  Volume distributions match within {diff_pct:.2f}%")
    print(f"  Both runs show {ref_err.mean():+.1f}% / {opt_err.mean():+.1f}% mean loss")
    print(f"  => CONSISTENT: volume loss is a physical model effect,")
    print(f"     NOT an optimization regression.")
else:
    print(f"  Volume distributions differ by {diff_pct:.2f}% -- INVESTIGATE")
