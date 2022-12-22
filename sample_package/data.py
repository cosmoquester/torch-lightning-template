import torch


class SampleDataset(torch.utils.data.Dataset):
    """Sample Dataset for testing

    Attributes:
        input_dimension: input dimension for linear layer
        num_classes: the number of classes
    """

    def __init__(self, input_dimension: int, num_classes: int):
        super().__init__()

        self.input_dimension = input_dimension
        self.num_classes = num_classes

    def __len__(self) -> int:
        return 10000

    def __getitem__(self, index: int) -> torch.Tensor:
        x = torch.randn(self.input_dimension)
        label = torch.randint(0, self.num_classes, [])
        return {"input": x, "label": label}
