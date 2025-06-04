import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import torch.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):

    def __init__(self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512]):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.n_classes = out_channels
        
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature
        
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(in_channels=feature*2, out_channels=feature))
        
        self.bottelneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottelneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1) # batch, channel, height, width

            x = self.ups[idx + 1](concat_skip)
        
        return self.final_conv(x)
    


def test():
    x = torch.randn((1, 2, 128, 128))
    model = UNET(in_channels=2, out_channels=1)

    predict = model(x)
    print(x.shape)
    print(predict.shape)
    assert x.shape == predict.shape



def test_LSTM():
    x = torch.randn((1, 100, 1, 1, 2))

    model = LSTMRegressor(input_dim=2, hidden_dim=64, spatial_size = 1)

    predict = model(x)
    print(x.shape)
    print(predict)
    # assert x.shape == predict.shape

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, spatial_size = 1):
        super().__init__()
        self.spatial_size = spatial_size
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(spatial_size,spatial_size,2)),  # (B*T, 1, 1, 1, 2) -> (B*T, 2, 1, 1, 1)
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x): 
        B, T, D, H, W = x.shape
        # x = x.unsqueeze(2)  # New shape: (1, 100, 1, spatial_size, spatial_size, 2)

        # #Combine batch and time
        # x = x.view(-1, 1, self.spatial_size, self.spatial_size, 2)  # Shape: (100, 1, spatial_size, spatial_size, 2)

        # x = self.encoder(x)             # -> (B*T, 2, spatial_size, spatial_size, 1)
        # print(x.shape)
        # sdf
        x = x.view(B, T, 2)             # Flatten back: (B, T, 8)

        out, _ = self.lstm(x)     # (B, 100, H)
        return self.fc(out)       # (B, 100, 1)



# if __name__ == "__main__":

#     test_LSTM()