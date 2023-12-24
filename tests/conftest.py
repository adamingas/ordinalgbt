import pytest
from ordinal_gbt.data import make_ordinal_classification

@pytest.fixture()
def df_ordinal_problem():
    return make_ordinal_classification(4,random_seed=42)