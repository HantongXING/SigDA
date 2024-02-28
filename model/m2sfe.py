import torch
import torch.nn as nn

__all__ = [
    "M2SFE", "M2SFE_medium", "M2SFE_tiny"
]

class M2SFE(nn.Module):
    def __init__(self):
        super(M2SFE, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=50,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=50, out_channels=256,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=512,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=512, out_channels=1024,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU())

        self.reconstructor = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1024, out_channels=512,
                               kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=512, out_channels=256,
                               kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=50,
                               kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=50, out_channels=2,
                               kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(2))

        self.cnn_mapping = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=512, out_channels=256,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=128,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=50,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(50),
            nn.LeakyReLU())

        self.rnn_mapping = nn.LSTM(128, 128, num_layers=2, batch_first=True)

        self.classifer = nn.Sequential(
            nn.Linear(in_features=6400, out_features=2048),
            nn.Dropout(0.6),
            nn.LeakyReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(0.6),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.Dropout(0.6),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=11))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        shallow_feature = self.feature_extractor(x)
        cons_input = self.reconstructor(shallow_feature)

        cnn_feature = self.cnn_mapping(shallow_feature)
        rnn_feature, _ = self.rnn_mapping(cnn_feature)
        rnn_feature = rnn_feature.contiguous().view(rnn_feature.size(0), -1)
        logits = self.classifer(rnn_feature)
        return logits, rnn_feature, cons_input


class M2SFE_tiny(nn.Module):
    def __init__(self):
        super(M2SFE_tiny, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )

        self.reconstructor = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32,
                               kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16,
                               kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=2,
                               kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(2))

        self.cnn_mapping = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=2,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(2),
            nn.LeakyReLU())

        self.rnn_mapping = nn.LSTM(
            2, 128, num_layers=1, batch_first=True, dropout=1)

        self.classifer = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.Dropout(0.6),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=11))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        shallow_feature = self.feature_extractor(x)
        cons_input = self.reconstructor(shallow_feature)

        cnn_feature = self.cnn_mapping(shallow_feature)
        _, (h, c) = self.rnn_mapping(cnn_feature.transpose(1, 2))

        logits = self.classifer(h[-1])
        return logits, h[-1], cons_input


class M2SFE_medium(nn.Module):
    def __init__(self):
        super(M2SFE_medium, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU())

        self.reconstructor = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=128,
                               kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=64,
                               kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=32,
                               kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=2,
                               kernel_size=3, padding=1, output_padding=0, stride=1),
            nn.BatchNorm1d(2))

        self.cnn_mapping = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=128,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=64,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=50,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(50),
            nn.LeakyReLU())

        self.rnn_mapping = nn.LSTM(
            128, 128, num_layers=2, batch_first=True, dropout=0.8)

        self.classifer = nn.Sequential(
            nn.Linear(in_features=6400, out_features=2048),
            nn.Dropout(0.6),
            nn.LeakyReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(0.6),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.Dropout(0.6),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=11))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        shallow_feature = self.feature_extractor(x)
        cons_input = self.reconstructor(shallow_feature)

        cnn_feature = self.cnn_mapping(shallow_feature)
        rnn_feature, _ = self.rnn_mapping(cnn_feature)
        rnn_feature = rnn_feature.contiguous().view(rnn_feature.size(0), -1)
        logits = self.classifer(rnn_feature)
        return logits, rnn_feature, cons_input


if __name__ == '__main__':

    from thop import profile
    from thop import clever_format

    model = M2SFE()

    myinput1 = torch.zeros((1, 2, 128)).cuda()
    flops, params = profile(model.cuda(), inputs=(myinput1,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops', flops)
    print('params', params)
