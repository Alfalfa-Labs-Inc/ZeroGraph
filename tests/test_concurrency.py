from diffusionblocks_independent.concurrency import StreamInterval, summarize_intervals


def test_summarize_intervals_measures_overlap():
    report = summarize_intervals(
        [
            StreamInterval(0, 0.0, 2.0),
            StreamInterval(1, 1.0, 3.0),
            StreamInterval(2, 3.0, 4.0),
        ]
    )
    assert report["union_seconds"] == 4.0
    assert report["summed_interval_seconds"] == 5.0
    assert report["overlapped_union_seconds"] == 1.0
    assert report["overlap_fraction_of_union"] == 0.25
    assert report["average_active_intervals"] == 1.25
    assert report["maximum_active_intervals"] == 2


def test_summarize_intervals_rejects_invalid_values():
    import pytest

    with pytest.raises(ValueError):
        summarize_intervals([])
    with pytest.raises(ValueError):
        summarize_intervals([StreamInterval(0, 2.0, 1.0)])
