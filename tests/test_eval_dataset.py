from support_rag import config
from support_rag.eval_dataset import EVAL_DATASET


def test_eval_set_size():
    assert 12 <= len(EVAL_DATASET) <= 15


def test_entry_shape_and_grounding():
    valid_docs = {p.stem for p in config.KB_DIR.glob("*.md")}
    for row in EVAL_DATASET:
        assert row["inputs"]["question"].strip()
        facts = row["expectations"]["expected_facts"]
        assert isinstance(facts, list) and facts and all(f.strip() for f in facts)
        assert row["expectations"]["source_doc"] in valid_docs
