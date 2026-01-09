import argparse
from pathlib import Path
from typing import Iterable

from coin_counter_pipeline import list_image_files


def collect_filenames(image_dir: Path, lowercase: bool) -> Iterable[str]:
    for path in list_image_files(image_dir):
        try:
            rel_path = path.relative_to(image_dir)
        except ValueError:
            rel_path = path
        name = rel_path.as_posix()
        yield name.lower() if lowercase else name


def write_in_file(output_path: Path, filenames: Iterable[str]) -> None:
    filenames = list(filenames)
    lines = [str(len(filenames))]
    lines.extend(filenames)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="依影像資料夾自動產生 in.txt")
    parser.add_argument("--image-dir", type=Path, required=True, help="影像資料夾路徑")
    parser.add_argument("--output-file", type=Path, default=Path("in.txt"), help="輸出檔案名稱 (預設 in.txt)")
    parser.add_argument("--lowercase", action="store_true", help="將檔名轉成小寫 (部分評測需要)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_dir.exists():
        raise FileNotFoundError(f"找不到影像資料夾: {args.image_dir}")
    filenames = list(collect_filenames(args.image_dir, args.lowercase))
    if not filenames:
        raise ValueError(f"{args.image_dir} 中沒有找到影像檔案。")
    write_in_file(args.output_file, filenames)
    print(f"已輸出 {len(filenames)} 筆資料至 {args.output_file}")


if __name__ == "__main__":
    main()
