def ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.res_tower = torch.nn.Sequential(
            torch.nn.Conv2D(18, 256, kernel_size=3, stride=1),
            torch.nn.BatchNorm2D(256),
            torch.nn.ReLU(),
            torch.nn.Conv2D(256, 256, kernel_size=3, stride=1),
            torch.nn.BatchNorm2D(256),
            torch.nn.ReLU(),
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2D(256, 1, kernel_size=1, stride=1),
            torch.nn.BatchNorm2D(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 4 * 1, 1),
            torch.nn.Tanh(),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2D(256, 256, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2D(256),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(5 * 5 * 256, 5000),
            torch.nn.ReLU(),
            torch.nn.Linear(5000, 4672),
            torch.nn.Softmax(dim=1)
        )
