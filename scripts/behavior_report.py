"""Summarize raw observations separately from trusted decisions."""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(directory):
    counts = {
        name: Counter()
        for name in (
            "raw_eye_state",
            "raw_mouth_state",
            "raw_orientation_indicator",
            "authenticity",
            "primary_status",
            "observation_reason",
        )
    }
    maximum = dict(raw_closed_seconds=0.0, raw_mouth_seconds=0.0)
    observations = 0
    with (directory / "persons.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            observations += 1
            for name, counter in counts.items():
                counter[row.get(name, "not_recorded")] += 1
            for name in maximum:
                maximum[name] = max(maximum[name], row.get(name, 0))
    return dict(
        observations=observations,
        counts={k: dict(v) for k, v in counts.items()},
        maximum=maximum,
        validated_accuracy=False,
        policy="raw_geometry_is_not_verified_identity_or_mental_state",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))


if __name__ == "__main__":
    main()
