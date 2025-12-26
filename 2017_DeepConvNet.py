class DeepConvNet(nn.Module):
    def __init__(self, num_classes=4, chans=64, samples=1250):
        super(DeepConvNet, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=1, padding=0),
            nn.Conv2d(25, 25, kernel_size=(chans, 1), stride=1, padding=0),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=1, padding=0),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=1, padding=0),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=1, padding=0),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )

        # FC Layer with dynamically computed input size
        self.classifier = nn.Linear(self._compute_flatten_dim(chans, samples), num_classes)

    def _compute_flatten_dim(self, chans, samples):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            x = self.conv_block1(dummy)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, chans, samples)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x