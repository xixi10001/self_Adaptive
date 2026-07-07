from pathlib import Path


def test_distance_field_residue_removed_from_runtime_sources():
    source_dir = Path(__file__).resolve().parent
    source_files = [
        source_dir / "信息转换.py",
        source_dir / "rl_environment_baseline.py",
    ]
    forbidden_terms = [
        "".join(("s", "d", "f")),
        "distance_transform_" + "edt",
    ]

    hits = []
    for source_file in source_files:
        text = source_file.read_text(encoding="utf-8")
        lowered = text.lower()
        for term in forbidden_terms:
            if term in lowered:
                hits.append(f"{source_file.name}: {term}")

    assert not hits
