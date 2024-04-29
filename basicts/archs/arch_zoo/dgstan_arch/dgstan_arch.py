import torch
from torch import nn
import torch.nn.functional as F
import math
from .gcn import DGCN, GCN_1, MultiLayerPerceptron


class DGSTAN(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.hidden_dim = model_args["hidden_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.hidden_dim_1 = model_args["hidden_dim_1"]
        self.output_len = model_args["output_len"]
        self.layer = model_args["layer"]
        self.integration = model_args["integration"]
        self.dropout_rate = model_args["dropout_rate"]
        self.adj_mx = model_args["adj_mx"]
        self.gtclayer = model_args["gtclayer"]
        # Multi-Scale Residual Graph Convolution
        self.GCN_1 = GCN_1(self.input_len, self.hidden_dim, self.output_len, self.dropout_rate)
        self.Multi_Scale_Residual_Graph_Convolution = Multi_Scale_Residual_Graph_Convolution(self.input_len,
                                                                                             self.hidden_dim,
                                                                                             self.output_len,
                                                                                             self.dropout_rate,
                                                                                             self.num_nodes,
                                                                                             self.adj_mx, self.layer)
        self.mlp = MultiLayerPerceptron(int(self.output_len * (self.layer + 1)), self.hidden_dim_1, self.output_len)

        # Gated Temporal Convolution
        self.gtc_layers = nn.ModuleList(
            [Gated_Temporal_Convolution(in_channels=1, out_channels=2, kernel_size=(3, 1), padding=(1, 0)) for _ in range(self.gtclayer)])


        # Spatial-Temporal Attention
        self.sa = SpatialAttention(self.dropout_rate, self.integration)
        self.ta = TemporalAttention(channel=self.output_len, reduction=1)
        self.t = torch.nn.Parameter(torch.rand(self.num_nodes, self.output_len))
        self.s = torch.nn.Parameter(torch.rand(self.num_nodes, self.output_len))
        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.output_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # prepare data
        input_data = history_data[..., range(self.input_dim)]
        self.adj_mx = torch.Tensor(self.adj_mx)
        self.adj_mx = self.adj_mx.to(input_data.device).squeeze(0)
        self.adj_mx = DGSTAN.process_graph(self.adj_mx)
        input_data = input_data.squeeze(3).transpose(1, 2).contiguous()

        # Multi-Scale Residual Graph Convolution
        output, input = self.GCN_1(input_data, self.adj_mx)
        out = output
        output, input, out = self.Multi_Scale_Residual_Graph_Convolution(output, input, out)
        out = out.unsqueeze(3).permute(0, 2, 1, 3)
        out = self.mlp(out).transpose(1, 2)

        # Gated Temporal Convolution
        for Gated_Temporal_Convolution in self.gtc_layers:
            out = Gated_Temporal_Convolution(out)

        # Spatial-Temporal Attention
        out1 = self.sa(out, self.adj_mx)  # Spatial Attention
        out = out.transpose(1, 2)
        out = self.ta(out)  # Temporal Attention
        out = out.transpose(1, 2)
        out = out.squeeze(3)
        out = self.t * out + self.s * out1
        out = out.unsqueeze(3).permute(0, 2, 1, 3)
        # regression
        prediction = self.regression_layer(out)
        return prediction

    def process_graph(graph_data):  # \hat A = D_{-1/2}*A*D_{-1/2}
        N = graph_data.size(0)
        matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)
        graph_data = graph_data + matrix_i
        degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)
        degree_matrix = torch.pow(degree_matrix, -0.5).flatten()
        degree_matrix[torch.isinf(degree_matrix)] = 0.
        degree_matrix = torch.diag(degree_matrix)
        out = graph_data.matmul(degree_matrix).transpose(0, 1).matmul(degree_matrix)
        return out  # A = D_{-1/2}*A*D_{-1/2}


class Multi_Scale_Residual_Graph_Convolution(nn.Module):
    def __init__(self, input_len, hidden_dim, output_len, dropout_rate, num_nodes, adj_mx, layer):
        super(Multi_Scale_Residual_Graph_Convolution, self).__init__()
        self.dgcn_layers = nn.ModuleList(
            [DGCN(input_len, hidden_dim, output_len, dropout_rate, num_nodes, adj_mx) for _ in range(layer)])

    def forward(self, output, input, out):
        for dgcn in self.dgcn_layers:
            output, input, out = dgcn(output, input, out)
        return output, input, out


class Gated_Temporal_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Gated_Temporal_Convolution, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.glu = torch.nn.GLU()

    def forward(self, out):
        out1 = out.permute(0, 3, 2, 1)
        out1 = self.conv(out1)
        out1 = out1.permute(0, 3, 2, 1)
        out = self.glu(out1)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, Gdropout_rate, integration):
        super(SpatialAttention, self).__init__()
        self.integration = integration
        self.sigmoid = nn.Sigmoid()
        if self.integration == "expand":
            self.linear_1 = nn.Linear(2, 1)
        elif self.integration == "linear":
            self.linear_1 = nn.Linear(2, 12)
        self.dropout = nn.Dropout(Gdropout_rate)
        self.act = nn.ReLU()

    def forward(self, x, graph_data):
        x = x.transpose(1, 2)
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        flow_x = torch.cat([max_result, avg_result], 1)
        flow_x = flow_x.transpose(1, 2).squeeze(3)
        x = x.transpose(1, 2).squeeze(3)
        output_1 = torch.matmul(graph_data, flow_x)
        output_1 = self.linear_1(output_1)
        output_1 = self.dropout(output_1)
        output_1 = self.act(output_1)
        output_2 = self.sigmoid(output_1)
        if self.integration == "expand":
            output_2 = output_2.expand_as(x)
        output_2 = output_2 * x
        return output_2


class TemporalAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(TemporalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

