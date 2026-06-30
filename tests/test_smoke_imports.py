import compare_retrievers


def test_compare_configs_have_no_splade():
    names = set(compare_retrievers.CONFIGS)
    assert names == {"dense", "sparse_bm25", "hybrid_bm25"}
    assert not any("splade" in n for n in names)
