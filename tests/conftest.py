import pytest

from costrouter.config import ModelConfig, Profile
from costrouter.data.loaders import gsm8k_row_to_query, mmlu_row_to_query
from costrouter.data.schema import Query


@pytest.fixture
def queries() -> list[Query]:
    gsm = [
        gsm8k_row_to_query(
            {"question": f"What is {i} + {i}?", "answer": f"... #### {2 * i}"}, i, "test"
        )
        for i in range(20)
    ]
    mmlu = [
        mmlu_row_to_query(
            {
                "question": f"Question {i}?",
                "subject": "astronomy" if i % 2 else "virology",
                "choices": ["w", "x", "y", "z"],
                "answer": i % 4,
            },
            i,
        )
        for i in range(20)
    ]
    return gsm + mmlu


@pytest.fixture
def fake_profile() -> Profile:
    return Profile(
        small=ModelConfig(
            name="fake-small", provider="fake", usd_in_per_m=0.05, usd_out_per_m=0.08, skill=0.5
        ),
        large=ModelConfig(
            name="fake-large", provider="fake", usd_in_per_m=0.59, usd_out_per_m=0.79, skill=0.9
        ),
    )
