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

    #model = LSTMRegressor(input_dim=2, hidden_dim=64, spatial_size = 1)
    model = TransformerModel()
    predict = model(x)
    print(x.shape)
    print(predict)
    # assert x.shape == predict.shape

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, spatial_size = 1):
        super().__init__()
        self.input_dim = input_dim
        self.spatial_size = spatial_size
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(spatial_size,spatial_size,2)),  # (B*T, 1, 1, 1, 2) -> (B*T, 2, 1, 1, 1)
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x): 
        B, T, D, H = x.shape
        # x = x.unsqueeze(2)  # New shape: (1, 100, 1, spatial_size, spatial_size, 2)

        # #Combine batch and time
        # x = x.view(-1, 1, self.spatial_size, self.spatial_size, 2)  # Shape: (100, 1, spatial_size, spatial_size, 2)

        # x = self.encoder(x)             # -> (B*T, 2, spatial_size, spatial_size, 1)
        # print(x.shape)
        # sdf
        x = x.view(B, T, self.input_dim)             # Flatten back: (B, T, 8)

        out, _ = self.lstm(x)     # (B, 100, H)
        return self.fc(out)       # (B, 100, 1)

class TransformerModel(nn.Module):
    def __init__(self, d_model=32, seq_len=100):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, T, 1, 1, 2)
        x = x.view(x.size(0), x.size(1), 1)             # (B, T, 2)
        x = self.input_proj(x)                          # (B, T, d_model)
        x = x + self.pos_encoding[:, :x.size(1), :]     # (B, T, d_model)
        out = self.encoder(x)                           # (B, T, d_model)
        return self.output_proj(out)                    # (B, T, 1)


class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: scalar -> latent
        self.encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        # Decoder: latent -> scalar
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):  # x: (N, 1)
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class ScalarSequenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output: raw logits
        )

    def forward(self, x):  # x: (B, T, 1)
        return self.net(x)  # returns logits: (B, T, 1)

class AttentionLSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):  # x: (batch, seq_len, 1)
        B, T, D, H = x.shape

        x = x.view(B, T, 1)             # Flatten back: (B, T, 8)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention weights
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        attended = attn_weights * lstm_out  # weighted sum
        output = self.fc(attended)  # (batch, seq_len, 1)
        return output

class MLP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # self.weights = nn.Parameter(torch.randn((in_channels)), requires_grad=True)
    
    def forward(self, x):
        x = self.mlp(x)
        return x 
#if __name__ == "__main__":

#     test_LSTM()