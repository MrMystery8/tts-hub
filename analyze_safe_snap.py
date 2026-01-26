import json
import os

path = "outputs/dashboard_runs/1769337794_17e145/metrics.jsonl"

print(f"{'STAGE':<10} {'EPOCH':<6} {'ACCEPT%':<8} {'PAIR_ACC%':<10} {'AUC':<6} {'DET_LOSS':<8}")
print("-" * 60)

if not os.path.exists(path):
    print("File not found")
    exit(1)

with open(path) as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get("type") == "probe":
                stage = d.get("stage", "")
                epoch = d.get("epoch", 0)
                
                # accept rate might be missing in early epochs if added late?
                # No, I added it before the run.
                acc = d.get("accept_rate_pos_at_fpr_1pct")
                if acc is None:
                    # Fallback to TPR if accept rate not explicitly logged in JSON (though I added it to report)
                    acc = d.get("tpr_at_fpr_1pct")
                
                pair = d.get("pair_acc_cls_cond_1pct")
                auc = d.get("mini_auc")
                
                acc_str = f"{acc*100:.1f}" if acc is not None else "-"
                pair_str = f"{pair*100:.2f}" if pair is not None else "-"
                auc_str = f"{auc:.3f}" if auc is not None else "-"
                
                print(f"{stage:<10} {epoch:<6} {acc_str:<8} {pair_str:<10} {auc_str:<6} {'-':<8}")
            
            elif d.get("type") == "epoch":
                 stage = d.get("stage", "")
                 epoch = d.get("epoch", 0)
                 loss = d.get("loss_det")
                 if loss is not None:
                      print(f"{stage:<10} {epoch:<6} {'':<8} {'':<10} {'':<6} {loss:.4f}")

        except Exception:
            pass
