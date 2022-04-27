import asyncio
import numpy as np
import tritonclient.http as httpclient
from typing import Iterable, List
from .models import ModelData


class Mediator:
    """
    Mediator between the bot client and TIS, that uses http protocol.

    None of the methods are thread safe. The object is inteded to be used
    by a single thread.

    """

    def __init__(
        self, client: httpclient.InferenceServerClient, max_requests: int
    ):
        self.client = client
        self.model_configs = []
        self.async_requests = []
        self.results_queue = asyncio.Queue(maxsize=max_requests)

    def infer(self, model_inputs: Iterable[ModelData]):
        """
        Asynchroniously sends http inference requests to TIS.

        Parameters
        ----------
        model_inputs : Iterable of ModelData objects

        """
        if not isinstance(model_inputs, Iterable):
            raise TypeError("Expected model_inputs to be Iterable")

        for model_input in model_inputs:
            if not isinstance(model_input, ModelData):
                raise TypeError(
                    "Expected model_inputs to be Iterable of ModelData objects"
                )

            config = model_input.model_config
            self.model_configs.append(config)

            if model_input.batched:
                server_input = [
                    httpclient.InferInput(
                        config.input_name,
                        model_input.data.shape,
                        config.data_type,
                    )
                ]
                server_input[0].set_data_from_numpy(
                    model_input.data, binary_data=True
                )
            else:
                server_input = [
                    httpclient.InferInput(
                        config.input_name,
                        [1, *model_input.data.shape],
                        config.data_type,
                    )
                ]
                server_input[0].set_data_from_numpy(
                    np.expand_dims(model_input.data, axis=0), binary_data=True
                )

            server_output = [
                httpclient.InferRequestedOutput(
                    config.output_name, binary_data=True
                )
            ]

            self.async_requests.append(
                self.client.async_infer(
                    model_name=config.model_name,
                    inputs=server_input,
                    outputs=server_output,
                )
            )

    def __get_result(self, index):
        result = self.async_requests[index].get_result()

        config = self.model_configs[index]
        self.results_queue.put_nowait(
            (index, result.as_numpy(config.output_name))
        )

    async def get_results(self) -> List[ModelData]:
        """
        Get results requested by infer method of the object.

        Returns
        -------
        results : list of ModelData objects containing received data

        """
        requests_num = len(self.async_requests)
        if requests_num == 0:
            return []

        loop = asyncio.get_running_loop()
        await asyncio.wait(
            [
                loop.run_in_executor(None, self.__get_result, i)
                for i in range(requests_num)
            ]
        )

        results = [None] * requests_num
        for _ in range(requests_num):
            index, data = self.results_queue.get_nowait()
            results[index] = ModelData(
                data=data,
                model_config=self.model_configs[index],
                batched=(data.shape[0] != 1),
            )
        return results
