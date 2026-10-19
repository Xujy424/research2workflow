from pathlib import Path
import zipfile
import tempfile
import shutil
import polars as pl


CIFTABLE_PATTERNS = {
    "szwt": "mdl_6_33_0",
    "szcj": "mdl_6_36_0",
    "sh": "mdl_4_24_0",
    "szshot": "mdl_6_28_0",
    "shshot": "MarketData",
}

L2DATA_PATH = Path(
    "Z:/stock/l2"
)

INPUT_DIR = L2DATA_PATH / "raw1"
OUTPUT_DIR = L2DATA_PATH / "raw"


def sink_csv(tmp_csv: Path, tmp_pq: Path):
    pl.scan_csv(tmp_csv, truncate_ragged_lines=True).sink_parquet(
        tmp_pq, compression="gzip",
    )


def convert_file(src: Path, dst: Path):
    """ZIP压缩CSV -> GZIP压缩Parquet"""

    if dst.exists():
        print(f"SKIP: {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_pq = dst.with_suffix(".tmp")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            if zipfile.is_zipfile(src):
                with zipfile.ZipFile(src) as zf:
                    members = [
                        info for info in zf.infolist()
                        if not info.is_dir()
                        and info.file_size > 0
                    ]

                    if not members:
                        raise ValueError("ZIP内没有有效文件")

                    # 将ZIP内CSV解压到临时目录
                    tmp_csv = tmp_dir / "data.csv"

                    with zf.open(members[0]) as fsrc:
                        with open(tmp_csv, "wb") as fdst:
                            shutil.copyfileobj(fsrc, fdst)

            else:
                tmp_csv = src

            # Lazy CSV -> GZIP Parquet
            sink_csv(tmp_csv, tmp_pq)

        # 写入成功后再替换正式文件
        tmp_pq.replace(dst)

        print(f"OK: {src.name} -> {dst}")

    except Exception as e:
        tmp_pq.unlink(missing_ok=True)
        print(f"ERROR: {src}: {e}")


def main():
    for date_dir in sorted(INPUT_DIR.iterdir()):

        if not date_dir.is_dir():
            continue

        date = date_dir.name

        if not (len(date) == 8 and date.isdigit()):
            continue

        for key, pattern in CIFTABLE_PATTERNS.items():

            matches = [
                p for p in date_dir.iterdir()
                if p.is_file()
                and p.name.startswith(f"{date}_{pattern}.csv")
            ]

            if not matches:
                print(f"MISSING: {date} {key}")
                continue

            if len(matches) > 1:
                print(f"WARNING: 多个匹配文件 {date} {key}")
                continue

            src = matches[0]

            dst = (
                OUTPUT_DIR
                / date
                / f"{key}.pq"
            )

            convert_file(src, dst)


if __name__ == "__main__":
    main()
