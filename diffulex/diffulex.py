from diffulex.config import Config
from diffulex.engine.dp_worker import DiffulexDPWorker
from diffulex.engine.tp_worker import DiffulexTPWorker

class Diffulex:
    def __new__(cls, model, **kwargs):
        data_parallel_size = kwargs.get('data_parallel_size', 1)
        if data_parallel_size > 1:
            return DiffulexDPWorker(model, **kwargs)
        return DiffulexTPWorker(model, **kwargs)