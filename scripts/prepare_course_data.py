from pathlib import Path
import argparse
import io
import zipfile


def extract_checked(archive, destination):
    for item in archive.infolist():
        path = destination / item.filename
        if not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Archive contains a path outside the destination")
        if "__MACOSX" not in Path(item.filename).parts and not Path(item.filename).name.startswith("."):
            archive.extract(item, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parents[1] / "data" / "course"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as outer:
        for name in ["4_ObjectCategories.zip", "15_ObjectCategories.zip", "queries.zip", "database.zip"]:
            if name in outer.namelist():
                with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                    extract_checked(inner, destination)
        for name in ["cameraman.tif", "mountain1.png", "mountain2.png", "query.jpg", "db.jpg"]:
            if name in outer.namelist():
                (destination / name).write_bytes(outer.read(name))
    print(f"Imported course images to {destination}")


if __name__ == "__main__":
    main()
