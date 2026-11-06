"""Verify and restore the exact binary files recorded by this evidence branch."""
import argparse
import bisect
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile


class SplitArchive(io.RawIOBase):
    def __init__(self, directory, parts):
        self.parts = [directory / part['file'] for part in parts]
        self.offsets = [0]
        for part in parts:
            self.offsets.append(self.offsets[-1] + part['bytes'])
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.offsets[-1] + offset
        if position < 0:
            raise ValueError('Negative archive position')
        self.position = position
        return position

    def read(self, size=-1):
        end = self.offsets[-1] if size < 0 else min(self.offsets[-1], self.position + size)
        blocks = []
        while self.position < end:
            index = bisect.bisect_right(self.offsets, self.position) - 1
            length = min(end, self.offsets[index + 1]) - self.position
            with self.parts[index].open('rb') as stream:
                stream.seek(self.position - self.offsets[index])
                block = stream.read(length)
            if len(block) != length:
                raise ValueError('Truncated archive part')
            blocks.append(block)
            self.position += length
        return b''.join(blocks)


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def safe_target(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):
        raise ValueError(f'Unsafe artifact path {relative}')
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Artifact path escaped destination {relative}')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', action='store_true', help='Restore archived files after verification')
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / 'artifacts'
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    overall = hashlib.sha256()
    for part in manifest['parts']:
        path = directory / part['file']
        if path.stat().st_size != part['bytes']:
            raise ValueError(f'Part size mismatch {path.name}')
        current = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                current.update(block)
                overall.update(block)
        if current.hexdigest() != part['sha256']:
            raise ValueError(f'Part checksum mismatch {path.name}')
    if overall.hexdigest() != manifest['archive_sha256']:
        raise ValueError('Archive checksum mismatch')
    root = args.destination.resolve()
    with zipfile.ZipFile(SplitArchive(directory, manifest['parts'])) as archive:
        for item in manifest['objects']:
            name = 'sha256/' + item['sha256']
            with archive.open(name) as stream:
                if digest(stream) != item['sha256']:
                    raise ValueError(f'Artifact checksum mismatch {name}')
            if args.extract:
                for relative in item['restore_paths']:
                    target = safe_target(root, relative)
                    if target.exists():
                        with target.open('rb') as stream:
                            if digest(stream) != item['sha256']:
                                raise ValueError(f'Existing file differs {relative}')
        restored = 0
        if args.extract:
            for item in manifest['objects']:
                for relative in item['restore_paths']:
                    target = safe_target(root, relative)
                    if target.exists():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + '.artifact-partial')
                    with archive.open('sha256/' + item['sha256']) as source, temporary.open('xb') as output:
                        for block in iter(lambda: source.read(1024 * 1024), b''):
                            output.write(block)
                    temporary.rename(target)
                    restored += 1
    print(json.dumps({'verified_parts': len(manifest['parts']), 'verified_unique_artifacts': len(manifest['objects']), 'restored_files': restored}, indent=2))


if __name__ == '__main__':
    main()
