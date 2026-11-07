import pytest

from text_summarization import config
from text_summarization.testing import build_tiny_seq2seq

ARTICLE = ("The mayor announced a new transport plan on Monday. The plan will add forty buses to the city network. "
           "Critics say the plan is too expensive. The weather was sunny and warm. "
           "The council will vote on the transport plan next week. Local shops expect more customers. "
           "The transport plan costs ten million dollars.")


@pytest.fixture(scope="session")
def sample_dir():
    assert (config.SAMPLE_DIR / "test.csv").exists(), "Run scripts/generate_sample_data.py"
    return config.SAMPLE_DIR


@pytest.fixture(scope="session")
def tiny_t5(tmp_path_factory):
    return str(build_tiny_seq2seq(tmp_path_factory.mktemp("tiny_t5"), "t5"))


@pytest.fixture(scope="session")
def tiny_bart(tmp_path_factory):
    return str(build_tiny_seq2seq(tmp_path_factory.mktemp("tiny_bart"), "bart"))


@pytest.fixture
def article():
    return ARTICLE


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Redirect every output path of the project into a temporary directory."""
    root = config.PROJECT_ROOT
    for name in ("MODELS_DIR", "REPORTS_DIR", "FIGURES_DIR", "METRICS_DIR", "PREDICTIONS_DIR", "README_PATH",
                 "PROCESSED_DIR"):
        monkeypatch.setattr(config, name, tmp_path / getattr(config, name).relative_to(root))
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    return tmp_path
