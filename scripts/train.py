"""Transfer learning klasifikasi (slide 8-12, 20-21) pada citra potongan basket.

Lima model torchvision (slide 14) x tiga mode (slide 21):
  feature : bobot ImageNet, hanya head baru yang dilatih
  partial : bobot ImageNet, blok akhir + head baru dilatih (LR blok 10x lebih kecil)
  scratch : bobot acak, semua layer dilatih

Contoh:
  python scripts/train.py --model efficientnet_b0 --modes feature partial scratch --seeds 0 1 2
  python scripts/train.py --model all                      # 5 model x 3 mode
  python scripts/train.py --smoke                          # uji alur tanpa bobot pretrained (bukan hasil)
"""
import argparse, copy, csv, json, os, random, time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

KELAS = ["backboard", "ball", "rim", "latar"]
MODELS = ["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18", "resnet50"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # statistik ImageNet (slide 13)
LR_HEAD, LR_BLOCK = 1e-3, 1e-4                              # slide 21


# --------------------------------------------------------------------------- model
def get_head(name, m):
    """Modul classifier akhir yang diganti (padanan `m.fc` pada slide 9)."""
    if name.startswith("resnet"):
        return m.fc
    if name.startswith("mobilenet"):
        return m.classifier[3]
    return m.classifier[1]                                   # efficientnet


def set_head(name, m, n):
    if name.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, n)
    elif name.startswith("mobilenet"):
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    else:
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)


def last_blocks(name, m):
    """Blok akhir yang dibuka pada mode partial (padanan `layer4` pada slide 21)."""
    if name.startswith("resnet"):
        return m.layer4
    start = {"mobilenet_v3_small": 9, "mobilenet_v3_large": 13, "efficientnet_b0": 7}[name]
    return m.features[start:]


def build(name, mode, n_class, pretrained=True):
    """Kembalikan (model, param_groups) sesuai mode."""
    use_w = pretrained and mode != "scratch"
    weights = models.get_model_weights(name).IMAGENET1K_V1 if use_w else None
    m = models.get_model(name, weights=weights)
    if mode != "scratch":
        for p in m.parameters():
            p.requires_grad = False                          # bekukan seluruh backbone
    set_head(name, m, n_class)                               # head baru: requires_grad True
    head = get_head(name, m)
    if mode == "feature":
        groups = [dict(params=list(head.parameters()), lr=LR_HEAD)]
    elif mode == "partial":
        blocks = last_blocks(name, m)
        for p in blocks.parameters():
            p.requires_grad = True
        groups = [dict(params=list(blocks.parameters()), lr=LR_BLOCK),
                  dict(params=list(head.parameters()), lr=LR_HEAD)]
    elif mode == "scratch":
        groups = [dict(params=list(m.parameters()), lr=LR_HEAD)]
    else:
        raise ValueError(mode)
    return m, groups


def set_train_mode(m):
    """Lapisan BatchNorm yang beku tetap eval() agar statistik ImageNet tidak berubah (slide 11)."""
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm) and \
                not any(p.requires_grad for p in mod.parameters(recurse=False)):
            mod.eval()


# --------------------------------------------------------------------------- data
class Crops(Dataset):
    def __init__(self, root, rows, tf):
        self.items = [(Image.open(os.path.join(root, r["nama_file"])).convert("RGB"),
                       KELAS.index(r["kelas"])) for r in rows]
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, y = self.items[i]
        return self.tf(img), y


def loaders(root, bs, workers, seed):
    rows = list(csv.DictReader(open(os.path.join(root, "metadata.csv"))))
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "valid", "test")}
    train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
                          T.RandomHorizontalFlip(),
                          T.ColorJitter(0.3, 0.3, 0.3, 0.05),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    g = torch.Generator(); g.manual_seed(seed)
    mk = lambda s, tf, shuf: DataLoader(Crops(root, by[s], tf), batch_size=bs, shuffle=shuf,
                                        num_workers=workers, generator=g if shuf else None)
    return mk("train", train_tf, True), mk("valid", eval_tf, False), mk("test", eval_tf, False)


# --------------------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(m, dl, dev, crit):
    m.eval()
    loss = n = ok = 0
    conf = np.zeros((len(KELAS), len(KELAS)), dtype=int)
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        out = m(x)
        loss += crit(out, y).item() * len(y); n += len(y)
        pred = out.argmax(1); ok += (pred == y).sum().item()
        for a, b in zip(y.tolist(), pred.tolist()):
            conf[a, b] += 1
    return loss / n, ok / n, conf


def run_one(args, name, mode, seed, dev):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tag = f"{name}_{mode}_s{seed}"
    out = os.path.join(args.out, tag); os.makedirs(out, exist_ok=True)
    wdir = os.path.join(args.weights_dir, tag); os.makedirs(wdir, exist_ok=True)

    tr, va, te = loaders(args.data, args.batch, args.workers, seed)
    m, groups = build(name, mode, len(KELAS), pretrained=not args.smoke)
    m.to(dev)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    n_all = sum(p.numel() for p in m.parameters())
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    hist, best, t0 = [], dict(acc=-1, loss=9e9, ep=0, state=None), time.time()
    for ep in range(1, args.epochs + 1):
        set_train_mode(m)
        tl = tn = tok = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            o = m(x); l = crit(o, y); l.backward(); opt.step()
            tl += l.item() * len(y); tn += len(y); tok += (o.argmax(1) == y).sum().item()
        sched.step()
        vl, vacc, _ = evaluate(m, va, dev, crit)
        hist.append(dict(epoch=ep, train_loss=tl / tn, train_acc=tok / tn, val_loss=vl, val_acc=vacc,
                         waktu_s=time.time() - t0))
        if vacc > best["acc"] or (vacc == best["acc"] and vl < best["loss"]):
            best = dict(acc=vacc, loss=vl, ep=ep, state=copy.deepcopy(m.state_dict()))
        print(f"[{tag}] ep {ep:02d} loss {tl/tn:.3f} acc {tok/tn:.3f} | val loss {vl:.3f} acc {vacc:.3f}", flush=True)
    train_time = time.time() - t0

    m.load_state_dict(best["state"])
    torch.save(best["state"], os.path.join(wdir, "best.pt"))
    tl_, tacc, conf = evaluate(m, te, dev, crit)

    accs = [h["val_acc"] for h in hist]
    first = next((h["epoch"] for h in hist if h["val_acc"] >= 0.9), None)
    stable = next((i + 1 for i in range(len(accs)) if all(a >= 0.9 for a in accs[i:])), None)
    summary = dict(model=name, mode=mode, seed=seed, epochs=args.epochs, batch=args.batch,
                   lr_head=LR_HEAD, lr_block=LR_BLOCK if mode == "partial" else None,
                   pretrained=(mode != "scratch" and not args.smoke), smoke=args.smoke,
                   params_total=n_all, params_trained=n_train,
                   best_val_acc=best["acc"], best_epoch=best["ep"], test_acc=tacc, test_loss=tl_,
                   epoch_first_val_ge_0_9=first, epoch_stable_val_ge_0_9=stable,
                   train_time_s=train_time, device=str(dev),
                   torch=torch.__version__, n_train=len(tr.dataset), n_val=len(va.dataset), n_test=len(te.dataset))
    with open(os.path.join(out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    np.savetxt(os.path.join(out, "confusion_test.csv"), conf, fmt="%d", delimiter=",",
               header=",".join(KELAS), comments="")
    print(f"[{tag}] val {best['acc']:.3f} @ep{best['ep']} | test {tacc:.3f} | {train_time:.0f}s | "
          f"dilatih {n_train:,}/{n_all:,} param", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0", help="|".join(MODELS) + "|all")
    ap.add_argument("--modes", nargs="+", default=["feature", "partial", "scratch"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--data", default="dataset_raw")
    ap.add_argument("--out", default="results")
    ap.add_argument("--weights-dir", default="runs")
    ap.add_argument("--smoke", action="store_true",
                    help="uji alur saja: tanpa bobot pretrained, 1 epoch; hasil BUKAN untuk laporan")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.out, args.weights_dir = 1, "smoke_results", "smoke_runs"
        print("MODE SMOKE: bobot pretrained tidak dipakai, hasil bukan untuk laporan.")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = MODELS if args.model == "all" else [args.model]
    for n in names:
        for mode in args.modes:
            for s in args.seeds:
                run_one(args, n, mode, s, dev)


if __name__ == "__main__":
    main()
