import pytest
import time
import asyncio
import numpy as np

from unittest.mock import Mock
from models.models import ModelConfig, ModelData
from models.mediator import Mediator
from datetime import datetime

model_config = ModelConfig("model_name")


@pytest.mark.parametrize(
    "model_inputs",
    [
        42,
        "safs",
        ["something"],
        ["something", "something"],
        [
            ModelData(np.arange(5, dtype=np.float32), model_config, False),
            "something",
        ],
    ],
)
def test_infer_incorrect_input(model_inputs):
    client = Mock()
    mediator = Mediator(client, max_requests=10)

    with pytest.raises(TypeError):
        mediator.infer(model_inputs)


def test_infer_incorrect_input_not_iterable():
    client = Mock()
    mediator = Mediator(client, max_requests=10)

    data = np.arange(100, dtype=np.float32)
    with pytest.raises(TypeError):
        mediator.infer(ModelData(data, model_config, False))


def test_infer_correct_input():
    client = Mock()
    client.async_infer.return_value = None

    mediator = Mediator(client, max_requests=10)

    data1 = np.arange(1000, dtype=np.float32).reshape((100, 10))
    data2 = np.arange(500, dtype=np.float32).reshape((5, 100))
    model_inputs = [
        ModelData(data1, model_config, False),
        ModelData(data2, model_config, False),
    ]
    mediator.infer(model_inputs)

    assert mediator.model_configs == [model_config, model_config]
    assert mediator.async_requests == [None, None]
    assert client.async_infer.call_count == 2


def test_get_results_no_infer():
    client = Mock()

    mediator = Mediator(client, max_requests=10)

    async def check_results():
        result = await mediator.get_results()
        assert result == []

    asyncio.run(check_results())


class AsyncRequestMock:
    def __init__(self, delay: int, result: np.ndarray):
        self.delay = delay
        self.request_result = Mock()
        self.request_result.as_numpy.return_value = result

    def get_result(self):
        time.sleep(self.delay)
        return self.request_result


def test_get_results_with_infer():
    data = np.arange(500, dtype=np.float32).reshape((5, 100))
    expected_data = np.arange(5).reshape(5, 1)

    model_inputs = [ModelData(data, model_config, True)]
    expected_result = [ModelData(expected_data, model_config, True)]

    client = Mock()

    async def check_results():
        client.async_infer.return_value = AsyncRequestMock(1, expected_data)

        mediator = Mediator(client, max_requests=10)
        mediator.infer(model_inputs)

        result = await mediator.get_results()
        assert expected_result == result

    asyncio.run(check_results())


def test_parallel_get_result():
    data = np.arange(500, dtype=np.float32).reshape((5, 100))
    expected_data = np.arange(5).reshape(5, 1)

    model_inputs = [
        ModelData(data, model_config, True),
        ModelData(data, model_config, True),
        ModelData(data, model_config, True),
    ]
    expected_result = [
        ModelData(expected_data, model_config, True),
        ModelData(expected_data, model_config, True),
        ModelData(expected_data, model_config, True),
    ]

    client = Mock()

    async def check_results():
        client.async_infer.return_value = AsyncRequestMock(2, expected_data)

        mediator = Mediator(client, max_requests=10)
        mediator.infer(model_inputs)

        result = await mediator.get_results()
        assert expected_result == result

    start = datetime.now()
    asyncio.run(check_results())
    duration = (datetime.now() - start).total_seconds()
    assert duration < 3.0
