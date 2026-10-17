from pathlib import Path
import codecs
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
    "/data/shanghai/xujiayi/workflow/data/stock/l2"
)

INPUT_DIR = L2DATA_PATH / "raw0"
OUTPUT_DIR = L2DATA_PATH / "pq"


def sink_csv(tmp_csv: Path, tmp_pq: Path, tmp_dir: Path):
    """识别 Unicode BOM；无 BOM 时优先 UTF-8，再严格尝试 GB18030。"""
    def sink(path):
        pl.scan_csv(path, truncate_ragged_lines=True).sink_parquet(
            tmp_pq, compression="gzip",
        )

    with tmp_csv.open("rb") as source:
        header = source.read(64)
    encoding = None
    # UTF-32 LE 的 BOM 以 UTF-16 LE BOM 开头，必须先检查 UTF-32。
    for bom, candidate in (
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    ):
        if header.startswith(bom):
            encoding = candidate
            break
    if encoding is None:
        try:
            sink(tmp_csv)
            return
        except pl.exceptions.ComputeError as exc:
            if "invalid utf-8" not in str(exc).lower():
                raise
            tmp_pq.unlink(missing_ok=True)
        encoding = "gb18030"

    try:
        utf8_csv = tmp_dir / "utf8.csv"
        # 分块转码以限制内存；严格解码，不静默丢弃或替换非法字节。
        with tmp_csv.open("r", encoding=encoding, newline="") as source:
            with utf8_csv.open("w", encoding="utf-8", newline="") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    except UnicodeDecodeError as exc:
        bad_bytes = exc.object[max(0, exc.start - 16):exc.end + 16]
        raise ValueError(
            f"Cannot decode CSV as {encoding}: {exc}; "
            f"header_hex={header.hex(' ')}; "
            f"error_context_hex={bad_bytes.hex(' ')}. "
            "Inspect ZIP member format/encoding; no bytes were replaced."
        ) from exc
    print(f"ENCODING: {tmp_csv.name}: {encoding} -> UTF-8")
    sink(utf8_csv)


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
            sink_csv(tmp_csv, tmp_pq, tmp_dir)

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
