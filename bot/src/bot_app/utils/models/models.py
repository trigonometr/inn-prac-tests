from dataclasses import dataclass
import numpy as np


@dataclass
class ModelConfig:
    """
    ModelConfig is used to describe model configuration attributes

    Parameters
    ----------
    model_name : the name of the model served by TIS

    input/output_name : input/output name defined in config.pbtxt for TIS

    input/output_data_type : input/output data type of the model defined
        in config.pbtxt.
    """

    model_name: str
    input_name: str = "INPUT__0"
    output_name: str = "OUTPUT__0"
    data_type: str = "FP32"


@dataclass
class ModelData:
    """
    Wraps the input/output model data with ModelConfig

    Parameters
    ----------
    data : data received from or to be sent to TIS

    model_config : model configuration, see ModelConfig

    batched : if the np.ndarray given in the data parameter represents a batch

    """

    data: np.ndarray
    model_config: ModelConfig
    batched: bool = False
