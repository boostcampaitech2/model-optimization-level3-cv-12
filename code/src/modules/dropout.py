from torch import nn

from src.modules.base_generator import GeneratorAbstract

class DropoutGenerator(GeneratorAbstract):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def base_module(self) -> nn.Module:
        """Base module."""
        return getattr(nn, f"{self.name}")
    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        # error: Value of type "Optional[List[int]]" is not indexable
        return self.in_channel

    def __call__(self, repeat: int = 1):
        module = (
            [self.base_module(*self.args) for _ in range(repeat)]
            if repeat > 1
            else self.base_module(*self.args)
        )
        return self._get_module(module)