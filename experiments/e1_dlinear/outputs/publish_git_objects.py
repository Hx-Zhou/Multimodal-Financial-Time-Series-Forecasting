"""Prepare GitHub Git-data API payloads from the exact committed Git objects."""
import json
import subprocess
from pathlib import Path


def git(*args):
    return subprocess.check_output(["git", *args])


root = Path(__file__).resolve().parent
head = git("rev-parse", "HEAD").decode().strip()
parent = git("rev-parse", "HEAD^").decode().strip()
tree = git("rev-parse", "HEAD^{tree}").decode().strip()
base_tree = git("rev-parse", "HEAD^^{tree}").decode().strip()
changed = [x.decode("utf-8") for x in git("diff-tree", "--no-commit-id", "--name-only", "-r", "-z", "HEAD").split(b"\0") if x]
if not changed or any(not p.startswith("experiments/e1_dlinear/") for p in changed):
    raise RuntimeError("unexpected changed-file scope")
objects = {}
for line in git("ls-tree", "-r", "-z", "HEAD").split(b"\0"):
    if line:
        meta, path = line.split(b"\t", 1)
        mode, kind, sha = meta.decode().split()
        objects[path.decode("utf-8")] = (mode, kind, sha)
request = "".join(objects[path][2] + "\n" for path in changed).encode()
raw = subprocess.run(["git", "cat-file", "--batch"], input=request, stdout=subprocess.PIPE, check=True).stdout
offset, entries = 0, []
for path in changed:
    end = raw.index(b"\n", offset)
    sha, kind, size = raw[offset:end].decode().split()
    size = int(size)
    content = raw[end + 1:end + 1 + size]
    offset = end + 1 + size + 1
    mode, expected_kind, expected_sha = objects[path]
    if kind != "blob" or expected_kind != "blob" or sha != expected_sha:
        raise RuntimeError("unexpected Git object")
    entries.append({"path": path, "mode": mode, "type": "blob", "content": content.decode("utf-8")})
tree_payload = {"base_tree": base_tree, "tree": entries}
identity = git("show", "-s", "--format=%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI", "HEAD").decode().strip().split("\0")
message = git("cat-file", "commit", "HEAD").split(b"\n\n", 1)[1].decode("utf-8")
commit_payload = {"message": message, "tree": tree, "parents": [parent],
                  "author": dict(zip(("name", "email", "date"), identity[:3])),
                  "committer": dict(zip(("name", "email", "date"), identity[3:]))}
for filename, value in (("pr-tree.json", tree_payload), ("pr-commit.json", commit_payload)):
    (root / filename).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"commit": head, "parent": parent, "tree": tree, "files": len(changed),
                  "payload_bytes": (root / "pr-tree.json").stat().st_size}))
