class EEGNet(nn.Module):
    def __init__(self, num_classes=4, chans=64, samples=125, dropout_rate=0.5, kern_length=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()

        # 첫 번째 합성곱 층 (Temporal Convolution)
        self.conv1 = nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # 깊이별 합성곱 층 (Depthwise Convolution)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # 분리형 합성곱 층 (Separable Convolution)
        self.separable_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # 완전연결층 (Fully Connected Layer)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * (samples // 32), num_classes)

    def forward(self, x):
        if x.ndim == 3:  # (batch, 64, 1250) 형태로 들어오면 변환
          x = x.unsqueeze(1)  # (batch, 1, 64, 1250)로 변환
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)  # ELU 활성화 추가

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)  # ELU 활성화 추가
        x = self.avgpool1(x)
        x = self.dropout1(x)

        x = self.separable_conv(x)
        x = self.bn3(x)
        x = self.elu(x)  # ELU 활성화 추가
        x = self.avgpool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x