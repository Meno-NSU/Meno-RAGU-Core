import pytest

from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from tests.ragu_fakes import pipeline


@pytest.fixture
def snapshot_pipeline():
    # the golden file was recorded with the legacy context style (DOCUMENT sections); it guards that path
    return pipeline(context_style="legacy"), PipelineRuntime.uniform(ModelRuntime("fake-model", "http://fake/v1"))


@pytest.fixture
def snapshot_question():
    return [ChatMessage(role="user", content="Какие факультеты есть в НГУ?")]
