"""GPU mapping, seeded by a table and corrected by observation."""
from auto_gpu4pyscf import gpumap
from auto_gpu4pyscf.gpumap import BUILT_IN, GpuMap


def test_builtin_is_used_when_nothing_was_learned(tmp_path):
    gpu_map = GpuMap(path=tmp_path / "gpu-map.json")
    assert gpu_map.lookup("a100") == (80, BUILT_IN)
    assert gpu_map.lookup("nvidia_a100-sxm4-40gb") == (80, BUILT_IN)
    assert gpu_map.lookup("something_new") == (None, "")


def test_learned_entry_wins_over_the_table(tmp_path):
    """The table guesses by substring, an observation is measured."""
    gpu_map = GpuMap(path=tmp_path / "gpu-map.json")
    gpu_map.record("a100", 90, source="probe on gpu")
    assert gpu_map.lookup("a100") == (90, "probe on gpu")


def test_record_reports_when_it_corrects_the_table(tmp_path):
    gpu_map = GpuMap(path=tmp_path / "gpu-map.json")
    assert gpu_map.record("a100", 80) is None, "agreement is not a correction"
    assert gpu_map.record("a100", 90) == 80, "should report the old guess"
    assert gpu_map.record("brand_new_card", 130) is None


def test_alias_and_full_name_are_both_learned(tmp_path):
    """sinfo and nvidia-smi name the same card differently."""
    gpu_map = GpuMap(path=tmp_path / "gpu-map.json")
    gpu_map.record_alias("gh200", "NVIDIA GH200 480GB", 90, source="probe on gpu")
    assert gpu_map.lookup("gh200")[0] == 90
    assert gpu_map.lookup("NVIDIA GH200 480GB")[0] == 90
    assert gpu_map.lookup("nvidia_gh200_480gb")[0] == 90


def test_learning_does_not_leak_between_similar_names(tmp_path):
    """a10 is a different card from a100, at a different capability."""
    gpu_map = GpuMap(path=tmp_path / "gpu-map.json")
    gpu_map.record("a100", 80, source="probe")
    assert gpu_map.lookup("a10") == (86, BUILT_IN)


def test_round_trip_through_the_file(tmp_path):
    path = tmp_path / "gpu-map.json"
    first = GpuMap(path=path)
    first.record("nvidia_h200", 90, source="probe on gpu-h200")
    first.save()
    second = GpuMap.load(path=path)
    assert second.lookup("nvidia_h200") == (90, "probe on gpu-h200")


def test_corrupt_or_missing_file_is_not_fatal(tmp_path):
    path = tmp_path / "gpu-map.json"
    assert GpuMap.load(path=path).learned == {}
    path.write_text("{oops")
    assert GpuMap.load(path=path).learned == {}
    path.write_text('{"gpus": {"x": {"capability": "eighty"}}}')
    assert GpuMap.load(path=path).learned == {}, "capabilities must be integers"


def test_unknown_models_are_the_probe_candidates(tmp_path):
    gpu_map = GpuMap(path=tmp_path / "gpu-map.json")
    names = ["a100", "some_new_card", "mi300x", "", "some_new_card"]
    assert gpu_map.unknown_models(names) == ["some_new_card"]
    gpu_map.record("some_new_card", 130)
    assert gpu_map.unknown_models(names) == []


def test_learn_local_records_and_only_writes_on_change(tmp_path, monkeypatch):
    path = tmp_path / "gpu-map.json"
    monkeypatch.setattr(gpumap, "_DEFAULT", GpuMap(path=path))
    gpumap.learn_local([("NVIDIA GeForce RTX 5070 Ti", "12.0")])
    assert gpumap.default().lookup("NVIDIA GeForce RTX 5070 Ti")[0] == 120
    assert path.is_file()

    written = path.stat().st_mtime_ns
    gpumap.learn_local([("NVIDIA GeForce RTX 5070 Ti", "12.0")])
    assert path.stat().st_mtime_ns == written, "unchanged map should not be rewritten"


def test_summary_mentions_both_sources(tmp_path):
    gpu_map = GpuMap(path=tmp_path / "gpu-map.json")
    gpu_map.record("a100", 80)
    assert "1 learned" in gpu_map.summary()
    assert "built-in" in gpu_map.summary()
