"""Downloads the SuiteSparse matrices used by the benchmarks."""

from __future__ import annotations
import io
import pathlib
import sys
import tarfile
import urllib.request
ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'data' / 'suitesparse_manifest.txt'
OUT_DIR = ROOT / 'data' / 'suitesparse'
URL_TMPL = 'https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz'

def ParseManifest(path: pathlib.Path):
    entries = []
    with path.open('r', encoding='utf-8') as fh:
        for raw in fh:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f'manifest line malformed: {raw!r}')
            name, group, dim, spd = (parts[0], parts[1], int(parts[2]), parts[3])
            entries.append((name, group, dim, spd))
    return entries

def MtxHeaderDim(path: pathlib.Path) -> int | None:
    try:
        with path.open('r', encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('%'):
                    continue
                rows, cols, _nnz = line.split()
                return int(rows)
    except OSError:
        return None
    return None

def FetchAndExtract(name: str, group: str, dim: int) -> str:
    dest = OUT_DIR / f'{name}.mtx'
    existing_dim = MtxHeaderDim(dest)
    if existing_dim == dim:
        return f'[skip] {name}.mtx already present ({dim}x{dim})'
    url = URL_TMPL.format(group=group, name=name)
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = resp.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tar:
        mtx_member = next((m for m in tar.getmembers() if m.name.endswith(f'{name}.mtx')), None)
        if mtx_member is None:
            raise RuntimeError(f'no {name}.mtx in archive from {url}')
        extracted = tar.extractfile(mtx_member)
        if extracted is None:
            raise RuntimeError(f'tarfile returned None for {name}.mtx')
        dest.write_bytes(extracted.read())
    got = MtxHeaderDim(dest)
    if got != dim:
        raise RuntimeError(f'{name}: header dim {got} != manifest dim {dim}')
    return f'[ok]   {name}.mtx ({dim}x{dim}) from {group}'

def Main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = ParseManifest(MANIFEST)
    print(f'Fetching {len(entries)} matrices into {OUT_DIR}')
    errors = 0
    for name, group, dim, _spd in entries:
        try:
            print(FetchAndExtract(name, group, dim))
        except Exception as exc:
            errors += 1
            print(f'[fail] {name}: {exc}', file=sys.stderr)
    return 1 if errors else 0
if __name__ == '__main__':
    sys.exit(Main())
