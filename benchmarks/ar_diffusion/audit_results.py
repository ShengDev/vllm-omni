# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Recompute the checked-in Wan benchmark from raw completion events."""

import argparse
import hashlib
import json
import math
import statistics
import zipfile


def audit(archive):
    requests = snapshots = 0
    rows = []
    expected = {
        1: "d37a9c999e612e6578aad0ff8fd715ba3d5abfa4a46994f698adb3a8df9396fe",
        2: "a9a53546cb02479643554cce66ee217c7c00d84af6cbb16a0ecde54b60c2f45d",
    }
    with zipfile.ZipFile(archive) as evidence:
        for case in ("k30-pr-head", "k15-pr-head"):
            summary = json.loads(evidence.read(f"{case}/summary.json"))
            for group in summary["groups"]:
                samples = []
                measured = []
                for name in evidence.namelist():
                    if name.startswith(f"{case}/g") and name.endswith(".json"):
                        request = json.loads(evidence.read(name))
                        if request["variant"] == group["variant"]:
                            measured.append(request)
                assert len(measured) == 4, "requires one warmup and three complete measured requests"
                assert sum(request["warmup"] for request in measured) == 1
                for request in sorted(measured, key=lambda value: value["repeat"]):
                    events = request["gpu_ms"]
                    assert len(events) == 128 and all(b > a for a, b in zip(events, events[1:]))
                    assert request["latent_shape"] == [1, 16, 384, 60, 104]
                    assert request["latent_sha256"] == expected[group["groups"]]
                    fps = 384000 / (events[95] - events[63])
                    assert math.isclose(fps, request["steady_fps"], abs_tol=1e-8)
                    if not request["warmup"]:
                        samples.append(fps)
                    requests += 1
                assert samples == group["fps_samples"]
                median = statistics.median(samples)
                assert math.isclose(median, group["median_fps"], abs_tol=1e-8)
                rows.append({"groups": group["groups"], "variant": group["variant"], "median_fps": median})
            manifest = json.loads(evidence.read(f"{case}/source_snapshot/manifest.json"))
            for name, digest in manifest.items():
                assert hashlib.sha256(evidence.read(f"{case}/source_snapshot/{name}")).hexdigest() == digest
                snapshots += 1
    return {"full_requests_checked": requests, "source_snapshots_checked": snapshots, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", help="Path to the checked-in evidence.zip")
    args = parser.parse_args()
    print(json.dumps(audit(args.evidence), indent=2))


if __name__ == "__main__":
    main()
