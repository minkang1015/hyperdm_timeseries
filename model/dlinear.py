import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer_layers import series_decomp

class DLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.dropout = configs.dropout # p=0.1

        # 정규화 레이어 추가 (LayerNorm)
        self.norm = nn.LayerNorm(self.channels)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # self._init_weights()

    def _init_weights(self):
        # Linear 레이어 가중치 초기화 (Xavier)
        if self.individual:
            for i in range(self.channels):
                nn.init.xavier_uniform_(self.Linear_Seasonal[i].weight)
                nn.init.zeros_(self.Linear_Seasonal[i].bias)
                nn.init.xavier_uniform_(self.Linear_Trend[i].weight)
                nn.init.zeros_(self.Linear_Trend[i].bias)
        else:
            nn.init.xavier_uniform_(self.Linear_Seasonal.weight)
            nn.init.zeros_(self.Linear_Seasonal.bias)
            nn.init.xavier_uniform_(self.Linear_Trend.weight)
            nn.init.zeros_(self.Linear_Trend.bias)

    def encoder(self, x):
        # 정규화 적용
        x = self.norm(x)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            seasonal_output = F.dropout(seasonal_output, p=self.dropout, training=self.training)
            trend_output = self.Linear_Trend(trend_init)
            trend_output = F.dropout(trend_output, p=self.dropout, training=self.training)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]