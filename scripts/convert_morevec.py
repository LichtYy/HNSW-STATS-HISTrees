#!/usr/bin/env python3

import h5py, numpy as np, struct, random, sys, os

SRC = sys.argv[1]; OUT = sys.argv[2]
NQ, SEED = 200, 42
os.makedirs(OUT, exist_ok=True)
f = h5py.File(SRC, "r")
keys = list(f.keys())
vkey = [k for k in keys if k.endswith("vector")][0]
V = np.asarray(f[vkey][:], dtype="float32")
N, dim = V.shape
ID = ("mid", "rid", "title", "_id")
num = [k for k in keys if f[k].ndim == 1 and np.issubdtype(f[k].dtype, np.floating)]
cat = [k for k in keys if f[k].ndim == 1 and f[k].dtype == object and not any(s in k for s in ID)]
short = lambda k: k.replace("train_", "")
print(f"{SRC}: N={N} dim={dim} numeric={[short(k) for k in num]} categorical={[short(k) for k in cat]}")

with open(f"{OUT}/base.fvecs", "wb") as o:
    hdr = struct.pack("<i", dim)
    for row in V:
        o.write(hdr); o.write(row.tobytes())

cols = num + cat
data = {k: f[k][:] for k in cols}
with open(f"{OUT}/attributes.csv", "w") as o:
    o.write(",".join(short(k) for k in cols) + "\n")
    for i in range(N):
        vals = []
        for k in cols:
            v = data[k][i]
            if isinstance(v, bytes): v = v.decode("utf-8", "replace").replace(",", ";")
            vals.append(str(v))
        o.write(",".join(vals) + "\n")
print(f"wrote attributes.csv ({len(cols)} cols)")

random.seed(SEED)
qidx = sorted(random.sample(range(N), min(NQ, N)))
rec = 4 + dim * 4
with open(f"{OUT}/base.fvecs", "rb") as fb, open(f"{OUT}/queries.fvecs", "wb") as fq:
    for i in qidx:
        fb.seek(i * rec); fq.write(fb.read(rec))
print(f"wrote {len(qidx)} queries")
