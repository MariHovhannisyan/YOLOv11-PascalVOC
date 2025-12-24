import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

csv_path = "yolo11m-finetune.csv"
df = pd.read_csv(csv_path)

df = df.dropna(subset=["epoch"]).copy()
x = df["epoch"].astype(int)

# 2) Extract metrics
precision = df["metrics/precision(B)"]
recall    = df["metrics/recall(B)"]
acc_map50 = df["metrics/mAP50(B)"]
map5095   = df["metrics/mAP50-95(B)"]

best_map50_i  = acc_map50.idxmax()
best_map5095_i = map5095.idxmax()

best_map50_epoch  = int(df.loc[best_map50_i, "epoch"])
best_map5095_epoch = int(df.loc[best_map5095_i, "epoch"])

best_map50  = float(acc_map50.loc[best_map50_i])
best_map5095 = float(map5095.loc[best_map5095_i])

out_dir = Path("viz_out")
out_dir.mkdir(exist_ok=True)


plt.figure(figsize=(12, 5))
plt.plot(x, precision, label="Precision")
plt.plot(x, recall,    label="Recall")
plt.plot(x, acc_map50, label="Accuracy (mAP@0.5)")
plt.plot(x, map5095,   label="mAP@0.5:0.95")

# mark best points
plt.scatter([best_map50_epoch], [best_map50], marker="P", s=120, label="Best Accuracy (mAP@0.5)")
plt.scatter([best_map5095_epoch], [best_map5095], marker="*", s=140, label="Best mAP@0.5:0.95")

# annotate
plt.annotate(f"Accuracy(mAP@0.5) = {best_map50:.3f}\n(Epoch {best_map50_epoch})",
             xy=(best_map50_epoch, best_map50),
             xytext=(best_map50_epoch, best_map50 + 0.03),
             arrowprops=dict(arrowstyle="->"))

plt.annotate(f"mAP@0.5:0.95 = {best_map5095:.3f}\n(Epoch {best_map5095_epoch})",
             xy=(best_map5095_epoch, best_map5095),
             xytext=(best_map5095_epoch, best_map5095 - 0.08),
             arrowprops=dict(arrowstyle="->"))

plt.title("YOLO Metrics (Accuracy) Curves")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(out_dir / "metrics_accuracy_curves.png", dpi=200)
plt.show()


normal_range_map50   = (0.70, 0.80)
normal_range_map5095 = (0.50, 0.57)

plt.figure(figsize=(10, 5))

labels = ["Accuracy (mAP@0.5)", "mAP@0.5:0.95"]
my_scores = [best_map50, best_map5095]
xpos = np.arange(len(labels))

# bars
bars = plt.bar(xpos, my_scores, width=0.55, label="My Best")

# shaded benchmark ranges
plt.fill_between([xpos[0]-0.45, xpos[0]+0.45],
                 [normal_range_map50[0]]*2, [normal_range_map50[1]]*2,
                 alpha=0.15, label="Normal Range")

plt.fill_between([xpos[1]-0.45, xpos[1]+0.45],
                 [normal_range_map5095[0]]*2, [normal_range_map5095[1]]*2,
                 alpha=0.15)

# text on bars
for i, v in enumerate(my_scores):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=12)

plt.xticks(xpos, labels)
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("My YOLO Results vs. Benchmark")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "benchmark.png", dpi=200)
plt.show()

plt.figure(figsize=(12, 5))

plt.plot(x, df["train/box_loss"], label="Train Box Loss")
plt.plot(x, df["val/box_loss"],   label="Val Box Loss")
plt.plot(x, df["train/cls_loss"], label="Train Cls Loss")
plt.plot(x, df["val/cls_loss"],   label="Val Cls Loss")

plt.plot(x, df["train/dfl_loss"], label="Train DFL Loss")
plt.plot(x, df["val/dfl_loss"],   label="Val DFL Loss")

plt.title("YOLO Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "loss_curves.png", dpi=200)
plt.show()

print(f"Saved images to: {out_dir.resolve()}")
