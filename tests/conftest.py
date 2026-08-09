import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def dataset():
    from diag_opt.data import load_dataset

    return load_dataset()
