class ShallowNet(nn.Module):
    def __init__(self, num_classes=4, chans=64, samples=1250):
        super(ShallowNet, self).__init__()
        
        self.temporal_conv = nn.Conv2d(1, 40, kernel_size=(1, 25), stride=1, bias=False)
        self.spatial_conv = nn.Conv2d(40, 40, kernel_size=(chans, 1), stride=1, groups=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(p=0.5)

        # ↓ flatten dim 정확히 계산
        flatten_dim = self._compute_flatten_dim(samples, chans)
        self.classifier = nn.Linear(flatten_dim, num_classes)
    
    def _compute_flatten_dim(self, samples, chans):
        # 정적인 크기를 예상하지 않고, dummy tensor로 실제 크기 측정
        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            out = self.temporal_conv(dummy)
            out = self.spatial_conv(out)
            out = self.batchnorm(out)
            out = out ** 2
            out = self.pool(out)
            out = torch.log(torch.clamp(out, min=1e-6))
            return out.numel()


    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)

        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.batchnorm(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x