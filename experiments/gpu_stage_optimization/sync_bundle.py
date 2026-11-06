"""Stream the authorized project/data to the user's own SSH workspace."""
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT.parent.parent / 'hs300_train_val'
REMOTE = '/home/liangzixuan/hs300_stageopt_20261101/project'
process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '243',
                            f'mkdir -p {REMOTE}; tar -xf - -C {REMOTE}'], stdin=subprocess.PIPE)
start = time.time()
with tarfile.open(fileobj=process.stdin, mode='w|') as archive:
    files = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.md')
             and not any(part in {'.git', '.venv', '__pycache__', 'outputs', 'node_modules'}
                         for part in p.relative_to(ROOT).parts)]
    for file in files:
        archive.add(file, arcname=file.relative_to(ROOT).as_posix(), recursive=False)
    for name in ('data/train.npz', 'data/val.npz', 'data/numerical', 'data/tabular', 'data/text/embeddings.npz'):
        print(f'Transferring {name}', flush=True)
        archive.add(PACKAGE / name, arcname='package/' + name)
process.stdin.close()
code = process.wait()
print(f'Transfer exit={code}, elapsed={time.time()-start:.1f}s', flush=True)
raise SystemExit(code)
