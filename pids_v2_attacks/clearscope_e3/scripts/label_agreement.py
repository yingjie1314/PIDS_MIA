"""Do the two arms agree on who is a member?

he_conti_v2 picks its threshold on the OTHER arm. That transfer is only meaningful if a node's
membership label is consistent across arms. On the no-frac arms agreement was 0.0000 (perfectly
role-swapped, so transfer is valid by inversion); on v2 CADETS it was 0.6482, which broke it.
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, "/data/pids_v2_attacks/clearscope_e3/scripts")
from ours_v2 import run_dir, node_ids, DETECTORS, SPLITS

def members(detector, arm, epoch=299):
    base = run_dir(detector, f"CLEARSCOPE_E3_V2_{arm}")
    out = {}
    for split in SPLITS:
        s = set()
        for p in glob.glob(os.path.join(base, split, f"model_epoch_{epoch}", "*.npz")):
            with np.load(p, allow_pickle=False) as z:
                n = node_ids(z, detector)
                if n is not None: s.update(n.tolist())
        out[split] = s
    return out["mia_train_in"], out["mia_heldout"]

rows = []
for d in DETECTORS:
    tm, tn = members(d, "TARGET")
    sm, sn = members(d, "SHADOW")
    t_all, s_all = tm | tn, sm | sn
    shared = t_all & s_all
    if not shared:
        print(f"  {d:22s} no shared nodes"); continue
    agree = sum(1 for n in shared if (n in tm) == (n in sm))
    m_vs_n = sum(1 for n in shared if n in tm and n in sn)   # member for target, non-member for shadow
    n_vs_m = sum(1 for n in shared if n in tn and n in sm)
    r = dict(detector=d, shared=len(shared), agreement=agree/len(shared),
             coverage=len(shared)/len(t_all), target_mem_shadow_non=m_vs_n, target_non_shadow_mem=n_vs_m)
    rows.append(r)
    print(f"  {d:22s} agreement {r['agreement']:.4f}  coverage {r['coverage']:.3f}  "
          f"shared {len(shared):,}  T-mem/S-non {m_vs_n:,}  T-non/S-mem {n_vs_m:,}", flush=True)
json.dump(rows, open("/data/pids_v2_attacks/clearscope_e3/results/label_agreement.json","w"), indent=2)
