import json
import struct

with open("models/gemma-3-1b/model.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    meta = json.loads(f.read(n))

names = [k for k in meta if k != "__metadata__"]
print("total tensors:", len(names))
for k in names[:14]:
    print(k, meta[k]["dtype"], meta[k]["shape"])
print("...")
for k in names[-4:]:
    print(k, meta[k]["dtype"], meta[k]["shape"])
