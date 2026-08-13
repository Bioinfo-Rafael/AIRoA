#!/usr/bin/env python3
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    official = (root / "external" / "PARC2026_pre" / "submission_template" / "policy_server.py").read_text()
    ours = (root / "submission" / "policy_server.py").read_text()
    class_marker = "class MyPolicy(BasePolicy):"
    suffix_marker = "# ============================================================\n# 以下は変更不可"
    if official.split(class_marker, 1)[0] != ours.split(class_marker, 1)[0]:
        raise SystemExit("policy_server.py changed before MyPolicy")
    if official.split(suffix_marker, 1)[1] != ours.split(suffix_marker, 1)[1]:
        raise SystemExit("policy_server.py changed after MyPolicy")
    print("Official PARC template outside MyPolicy: byte-identical")


if __name__ == "__main__":
    main()
