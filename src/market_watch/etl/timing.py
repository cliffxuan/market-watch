import gzip
import json
import zlib
from pathlib import Path
from timeit import timeit

PWD = Path(__file__).parent.absolute()
DATA_DIR = PWD.parent.parent.parent / "data"

with open(DATA_DIR / "info.json") as f:
    info = json.loads(f.read())


def read_zz(info):
    with open(DATA_DIR / "info.json.zz", "rb") as f:
        info = json.loads(zlib.decompress(f.read()))


def read_gz(info):
    with open(DATA_DIR / "info.json.gz", "rb") as f:
        info = json.loads(gzip.decompress(f.read()))


def main():
    print(
        timeit("read_zz(info)", setup="from __main__ import info, read_zz", number=20)
    )
    print(
        timeit("read_gz(info)", setup="from __main__ import info, read_gz", number=20)
    )


if __name__ == "__main__":
    main()
