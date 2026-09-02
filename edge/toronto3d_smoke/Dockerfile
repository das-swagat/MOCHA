FROM nvcr.io/nvidia/pytorch:25.05-py3-igpu

ENV TORCH_CUDA_ARCH_LIST=8.7
ENV CUMM_CUDA_ARCH_LIST=8.7
ENV MAX_JOBS=2
ENV PIP_NO_CACHE_DIR=1

RUN python3 -m pip install --no-cache-dir addict timm

RUN python3 -m pip install --no-cache-dir --no-build-isolation torch-scatter==2.1.2

WORKDIR /opt/src

RUN git clone --depth 1 https://github.com/FindDefinition/cumm.git && \
    python3 -m pip install --no-build-isolation -e /opt/src/cumm

RUN git clone --depth 1 https://github.com/traveller59/spconv.git && \
    python3 - <<'PY'
from pathlib import Path
p=Path("/opt/src/spconv/pyproject.toml")
s=p.read_text()
lines=[]
for line in s.splitlines():
    if "cumm" in line.lower() and "requires" not in line.lower():
        continue
    lines.append(line)
p.write_text("\n".join(lines)+"\n")
PY

RUN python3 -m pip install --no-build-isolation -e /opt/src/spconv && \
    python3 -c 'import spconv; print("spconv build/import PASS", spconv.__version__)'

RUN python3 - <<'PY'
from pathlib import Path
import subprocess
for name in ("cumm","spconv"):
    p=Path("/opt/src")/name
    sha=subprocess.check_output(["git","-C",str(p),"rev-parse","HEAD"],text=True).strip()
    print(name,sha)
PY
