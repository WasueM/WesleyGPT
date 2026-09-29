# Wesley wrote this
"""Models load on first use and the least recently used one is evicted, so the
server's memory holds at most `max_resident` models however many it offers."""
import pytest

from wesleygpt.runtime import NanochatRuntime

SPECS = [{"id": f"m{i}", "description": ""} for i in range(4)]


class CountingLoader:
    def __init__(self):
        self.loads = []

    def __call__(self, spec, device):
        self.loads.append(spec["id"])
        return object(), object()


def runtime(max_resident=2):
    loader = CountingLoader()
    return NanochatRuntime(SPECS, loader=loader, max_resident=max_resident), loader


def test_only_the_default_model_is_loaded_at_startup():
    rt, loader = runtime()
    assert loader.loads == ["m0"] and rt.resident() == ["m0"]


def test_a_model_loads_on_its_first_request_and_only_once():
    rt, loader = runtime()
    rt.ensure_loaded("m1")
    rt.ensure_loaded("m1")
    assert loader.loads == ["m0", "m1"]


def test_loading_past_the_limit_evicts_the_least_recently_used():
    rt, _ = runtime()
    rt.ensure_loaded("m1")
    rt.ensure_loaded("m2")
    assert sorted(rt.resident()) == ["m1", "m2"]


def test_using_a_model_protects_it_from_eviction():
    rt, _ = runtime()
    rt.ensure_loaded("m1")
    rt.ensure_loaded("m0")
    rt.ensure_loaded("m2")
    assert sorted(rt.resident()) == ["m0", "m2"]


def test_an_evicted_model_is_reloaded_when_asked_for_again():
    rt, loader = runtime(max_resident=1)
    rt.ensure_loaded("m1")
    rt.ensure_loaded("m0")
    assert loader.loads == ["m0", "m1", "m0"]


def test_max_resident_must_be_at_least_one():
    with pytest.raises(ValueError, match="max_resident"):
        runtime(max_resident=0)


def test_the_served_model_list_carries_each_summary():
    # /v1/models reads the summary from here; dropping it made every listing a 500.
    specs = [{"id": "a", "summary": "Short", "description": "Long"}]
    rt = NanochatRuntime(specs, loader=CountingLoader())
    assert rt.models == [{"id": "a", "summary": "Short", "description": "Long"}]


def test_a_spec_built_without_a_summary_lists_a_blank_one():
    # The evals and a Hugging Face release build their own specs, with no summary.
    assert NanochatRuntime(SPECS, loader=CountingLoader()).models[0]["summary"] == ""
