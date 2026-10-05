import sys, numpy as np, json
sys.path.insert(0, r"C:\Users\gaura\ml-projects\triage-iq-wt-sklearn\eval")
sys.path.insert(0, r"C:\Users\gaura\ml-projects\triage-iq-wt-sklearn")
from test_invariants import _compute_ece
ev = [json.loads(l) for l in open(r"C:\Users\gaura\ml-projects\triage-iq-wt-sklearn\eval\eval_set.jsonl")]
for ver in ("1.6.1", "1.7.2"):
    z = np.load(f"out_eq/equiv_{ver}.npz", allow_pickle=True)
    for slug, repo in (("microsoft_vscode", "microsoft/vscode"), ("kubernetes_kubernetes", "kubernetes/kubernetes")):
        cal = z[f"{slug}|cls_test|cal"]; cls = z[f"{slug}|cls_test|classes"]; y = z[f"{slug}|cls_test|y"]
        top3 = np.argsort(-cal, 1)[:, :3]
        t3 = np.mean([yy in set(cls[r]) for yy, r in zip(y, top3)])
        t1 = np.mean(cls[cal.argmax(1)] == y)
        yg = np.array([e["gold_component"] for e in ev if e["repo"] == repo])
        pe = z[f"{slug}|eval64|label"]; pc = z[f"{slug}|eval64|cal"]
        print(f"sklearn {ver} {slug}: test n={len(y)} top1={t1:.4f} top3={t3:.4f} | eval-set n={len(yg)} ECE(5bins)={_compute_ece(yg, pe, pc, n_bins=5):.4f}")
