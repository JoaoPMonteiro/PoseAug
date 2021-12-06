import torch
import torch.nn as nn
import numpy as np

'''
this folder and code is modified base on ST-GCN code,
https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks
the ST-GCN model for single frame setting.
Note: the original ST-GCN training strategy is carefully deasigned by two stages, 
with different loss, training parameter, learning rate. 
Here we only use it as a baseline model, which we follow the same training strategy with other three model for convenience.
'''

from models_baseline.models_st_gcn.st_gcn_utils.tgcn import st_gcn_ConvTemporalGraphical
from models_baseline.models_st_gcn.st_gcn_utils.graph_frames import st_gcn_Graph
from models_baseline.models_st_gcn.st_gcn_utils.graph_frames_withpool_2 import st_gcn_Graph_pool
from models_baseline.models_st_gcn.st_gcn_utils.st_gcn_non_local_embedded_gaussian import st_gcn_NONLocalBlock2D

inter_channels = [128, 128, 256]

fc_out = inter_channels[-1]
fc_unit = 512


class Model_defaultDropout(nn.Module):
    """

    Args:
        in_channels (int): Number of channels in the input data
        cat: True: concatinate coarse and fine features
            False: add coarse and fine features
        pad:


    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes for each frame,
            :math:`M_{in}` is the number of instance in a frame. (In this task always equals to 1)
    Return:
        out_all_frame: True: return all frames 3D results
                        False: return target frame result

        x_out: final output.

    """

    def __init__(self):
        super().__init__()

        # load graph
        self.momentum = 0.1
        self.in_channels = 2
        self.out_channels = 3
        self.layout = 'hm36_gt'
        self.strategy = 'spatial'
        self.cat = True
        self.inplace = True
        self.pad = 0  # for single frame

        # original graph
        self.graph = st_gcn_Graph(self.layout, self.strategy, pad=self.pad)
        # get adjacency matrix of K clusters
        #self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cuda()  # K, T*V, T*V
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cpu() #wally

        # pooled graph
        self.graph_pool = st_gcn_Graph_pool(self.layout, self.strategy, pad=self.pad)
        #self.A_pool = torch.tensor(self.graph_pool.A, dtype=torch.float32, requires_grad=False).cuda()
        self.A_pool = torch.tensor(self.graph_pool.A, dtype=torch.float32, requires_grad=False).cpu() #wally

        # build networks
        kernel_size = self.A.size(0)
        kernel_size_pool = self.A_pool.size(0)

        self.data_bn = nn.BatchNorm1d(self.in_channels * self.graph.num_node_each, self.momentum)

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(self.in_channels, inter_channels[0], kernel_size, residual=False),
            st_gcn(inter_channels[0], inter_channels[1], kernel_size),
            st_gcn(inter_channels[1], inter_channels[2], kernel_size),
        ))

        self.st_gcn_pool = nn.ModuleList((
            st_gcn(inter_channels[-1], fc_unit, kernel_size_pool),
            st_gcn(fc_unit, fc_unit, kernel_size_pool),
        ))

        self.conv4 = nn.Sequential(
            nn.Conv2d(fc_unit, fc_unit, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(fc_unit, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(fc_unit * 2, fc_out, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(fc_out, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.1)
        )

        self.non_local = st_gcn_NONLocalBlock2D(in_channels=fc_out * 2, sub_sample=False)

        # fcn for final layer prediction
        fc_in = inter_channels[-1] + fc_out if self.cat else inter_channels[-1]
        self.fcn = nn.Sequential(
            nn.Dropout(0.1, inplace=True),
            nn.Conv2d(fc_in, self.out_channels, kernel_size=1)
        )

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p, stride=None):
        if max(p) > 1:
            if stride is None:
                x = nn.MaxPool2d(p)(x)  # B x F x V/p
            else:
                x = nn.MaxPool2d(kernel_size=p, stride=stride)(x)  # B x F x V/p
            return x
        else:
            return x

    def forward(self, x, out_all_frame=False):

        # data normalization
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N, M, V, C, T
        x = x.view(N * M, V * C, T)

        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, 1, -1)  # (N * M), C, 1, (T*V)

        # forwad GCN
        gcn_list = list(self.st_gcn_networks)
        for i_gcn, gcn in enumerate(gcn_list):
            x, _ = gcn(x, self.A)  # (N * M), C, 1, (T*V)

        x = x.view(N, -1, T, V)  # N, C, T ,V

        # Pooling
        for i in range(len(self.graph.part)):
            num_node = len(self.graph.part[i])
            x_i = x[:, :, :, self.graph.part[i]]
            x_i = self.graph_max_pool(x_i, (1, num_node))
            x_sub1 = torch.cat((x_sub1, x_i), -1) if i > 0 else x_i  # Final to N, C, T, (NUM_SUB_PARTS)

        x_sub1, _ = self.st_gcn_pool[0](x_sub1.view(N, -1, 1, T * len(self.graph.part)),
                                        self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
        x_sub1, _ = self.st_gcn_pool[1](x_sub1, self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
        x_sub1 = x_sub1.view(N, -1, T, len(self.graph.part))

        x_pool_1 = self.graph_max_pool(x_sub1, (1, len(self.graph.part)))  # N, 512, T, 1
        x_pool_1 = self.conv4(x_pool_1)  # N, C, T, 1

        x_up_sub = torch.cat((x_pool_1.repeat(1, 1, 1, len(self.graph.part)), x_sub1), 1)  # N, 1024, T, 5
        x_up_sub = self.conv2(x_up_sub)  # N, C, T, 5

        # upsample
        #x_up = torch.zeros((N * M, fc_out, T, V)).cuda()
        x_up = torch.zeros((N * M, fc_out, T, V)).cpu() #wally
        for i in range(len(self.graph.part)):
            num_node = len(self.graph.part[i])
            x_up[:, :, :, self.graph.part[i]] = x_up_sub[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, num_node)

        # for non-local and fcn
        x = torch.cat((x, x_up), 1)
        x = self.non_local(x)  # N, 2C, T, V
        x = self.fcn(x)  # N, 3, T, V

        # output
        x = x.view(N, M, -1, T, V).permute(0, 2, 3, 4, 1).contiguous()  # N, C, T, V, M
        if out_all_frame:
            x_out = x
        else:
            x_out = x[:, :, self.pad].unsqueeze(2)
        return x_out


class Model(nn.Module):
    """

    Args:
        in_channels (int): Number of channels in the input data
        cat: True: concatinate coarse and fine features
            False: add coarse and fine features
        pad:


    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes for each frame,
            :math:`M_{in}` is the number of instance in a frame. (In this task always equals to 1)
    Return:
        out_all_frame: True: return all frames 3D results
                        False: return target frame result

        x_out: final output.

    """

    def __init__(self, dropout=0.1):
        super().__init__()

        # load graph
        self.momentum = 0.1
        self.in_channels = 2
        self.out_channels = 3
        self.layout = 'hm36_gt'
        self.strategy = 'spatial'
        self.cat = True
        self.inplace = True
        self.pad = 0  # for single frame

        # original graph
        self.graph = st_gcn_Graph(self.layout, self.strategy, pad=self.pad)
        # get adjacency matrix of K clusters
        #self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cuda()  # K, T*V, T*V
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cpu() #wally

        # pooled graph
        self.graph_pool = st_gcn_Graph_pool(self.layout, self.strategy, pad=self.pad)
        #self.A_pool = torch.tensor(self.graph_pool.A, dtype=torch.float32, requires_grad=False).cuda()
        self.A_pool = torch.tensor(self.graph_pool.A, dtype=torch.float32, requires_grad=False).cpu()

        # build networks
        kernel_size = self.A.size(0)
        kernel_size_pool = self.A_pool.size(0)

        self.data_bn = nn.BatchNorm1d(self.in_channels * self.graph.num_node_each, self.momentum)

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(self.in_channels, inter_channels[0], kernel_size, residual=False, dropout=dropout),
            st_gcn(inter_channels[0], inter_channels[1], kernel_size, dropout=dropout),
            st_gcn(inter_channels[1], inter_channels[2], kernel_size, dropout=dropout),
        ))

        self.st_gcn_pool = nn.ModuleList((
            st_gcn(inter_channels[-1], fc_unit, kernel_size_pool, dropout=dropout),
            st_gcn(fc_unit, fc_unit, kernel_size_pool, dropout=dropout),
        ))

        self.conv4 = nn.Sequential(
            nn.Conv2d(fc_unit, fc_unit, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(fc_unit, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(fc_unit * 2, fc_out, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(fc_out, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(dropout)
        )

        self.non_local = st_gcn_NONLocalBlock2D(in_channels=fc_out * 2, sub_sample=False)

        # fcn for final layer prediction
        fc_in = inter_channels[-1] + fc_out if self.cat else inter_channels[-1]
        self.fcn = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Conv2d(fc_in, self.out_channels, kernel_size=1)
        )

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p, stride=None):
        if max(p) > 1:
            if stride is None:
                x = nn.MaxPool2d(p)(x)  # B x F x V/p
            else:
                x = nn.MaxPool2d(kernel_size=p, stride=stride)(x)  # B x F x V/p
            return x
        else:
            return x

    # https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
    def get_im2col_indices(self, x_shape, field_height, field_width, padding=1, stride=1):
        # First figure out what the size of the output should be
        N, C, H, W = x_shape
        assert (H + 2 * padding - field_height) % stride == 0
        assert (W + 2 * padding - field_width) % stride == 0
        out_height = (H + 2 * padding - field_height) / stride + 1
        out_width = (W + 2 * padding - field_width) / stride + 1

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1).astype(int)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1).astype(int)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1).astype(int)

        return (k, i, j)

    # https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
    def im2col_indices(self, x, field_height, field_width, padding=1, stride=1):
        """ An implementation of im2col based on some fancy indexing """
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = self.get_im2col_indices(x.shape, field_height, field_width, padding,
                                     stride)

        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols

    #https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
    def col2im_indices(self, cols, x_shape, field_height=3, field_width=3, padding=1,
                       stride=1):
        """ An implementation of col2im based on fancy indexing and np.add.at """
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self.get_im2col_indices(x_shape, field_height, field_width, padding,
                                     stride)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]

    def test_maxPool(self, x_in):
        x = x_in.detach().numpy()
        #https://agustinus.kristia.de/techblog/2016/07/18/convnet-maxpool-layer/
        # [xx] Let say our input X is 5x10x28x28
        # [xx] Our pooling parameter are: size = 2x2, stride = 2, padding = 0
        # [xx] i.e. result of 10 filters of 3x3 applied to 5 imgs of 28x28 with stride = 1 and padding = 1

        # 1x256x1x5
        # size = (1,5), stride = (1,5)

        n = 1
        d = 256
        h = 1
        w = 5
        # First, reshape it to 50x1x28x28 to make im2col arranges it fully in column
        X_reshaped = x.reshape(n * d, 1, h, w)

        size = (1, 5)
        stride = 5
        # The result will be 4x9800
        # Note if we apply im2col to our 5x10x28x28 input, the result won't be as nice: 40x980
        X_col = self.im2col_indices(X_reshaped, h, w, padding=0, stride=stride)

        # Next, at each possible patch location, i.e. at each column, we're taking the max index
        max_idx = np.argmax(X_col, axis=0)

        # Finally, we get all the max value at each column
        # The result will be 1x9800
        out = X_col[max_idx, range(max_idx.size)]

        h_out = 1
        w_out = 1
        # Reshape to the output size: 14x14x5x10
        out = out.reshape(h_out, w_out, n, d)

        # Transpose to get 5x10x14x14 output
        out = out.transpose(2, 3, 0, 1)

        out_out = torch.from_numpy(out)

        return out_out.to('cpu')

    def forward(self, x, out_all_frame=False):

        # data normalization
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N, M, V, C, T
        x = x.view(N * M, V * C, T)

        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, 1, -1)  # (N * M), C, 1, (T*V)

        # forwad GCN
        gcn_list = list(self.st_gcn_networks)
        for i_gcn, gcn in enumerate(gcn_list):
            x, _ = gcn(x, self.A)  # (N * M), C, 1, (T*V)

        x = x.view(N, -1, T, V)  # N, C, T ,V

        # Pooling

        #for i in range(len(self.graph.part)):
        #    num_node = len(self.graph.part[i])
        #    #'''
        #    x_i = x[:, :, :, self.graph.part[i]] #wally
        #    x_i = self.graph_max_pool(x_i, (1, num_node)) #wally
        #    '''
        #    x_i = x[:, :, :, 0] #wally
        #    '''
        #    x_sub1 = torch.cat((x_sub1, x_i), -1) if i > 0 else x_i  # Final to N, C, T, (NUM_SUB_PARTS)

        x_i_0 = x[:, :, :, self.graph.part[0]]  # wally
        x_i_0 = self.graph_max_pool(x_i_0, (1, 3))  # wally
        x_sub1_alt = x_i_0
        x_i_1 = x[:, :, :, self.graph.part[1]]  # wally
        x_i_1 = self.graph_max_pool(x_i_1, (1, 3))  # wally
        x_sub1_alt = torch.cat((x_sub1_alt, x_i_1), -1)
        x_i_2 = x[:, :, :, self.graph.part[2]]  # wally
        x_i_2 = self.graph_max_pool(x_i_2, (1, 3))  # wally
        x_sub1_alt = torch.cat((x_sub1_alt, x_i_2), -1)
        x_i_3 = x[:, :, :, self.graph.part[3]]  # wally
        x_i_3 = self.graph_max_pool(x_i_3, (1, 3))  # wally
        x_sub1_alt = torch.cat((x_sub1_alt, x_i_3), -1)

        x_i_4 = x[:, :, :, self.graph.part[4]]  # wally
        #x_i_4 = self.graph_max_pool(x_i_4, (1, 5))  # wally

        #x_i_4_alt1 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))(x_i_4)
        x_i_4 = self.test_maxPool(x_i_4)

        #x_i_4 = torch.nn.functional.max_pool2d(x_i_4, (1, 5), (1, 5), 0, 1, False, False)
        #x_i_4 = x[:, :, :, [0]]

        x_sub1_alt = torch.cat((x_sub1_alt, x_i_4), -1)
        x_sub1 = x_sub1_alt


        #wally_0 = torch.eq(x_i_4_alt1, x_i_4_alt2)
        #wally_1 = wally_0.float()
        #wally_2 = torch.sum(wally_1)
        #wally_3 = torch.numel(wally_1)


        x_sub1, _ = self.st_gcn_pool[0](x_sub1.view(N, -1, 1, T * len(self.graph.part)),
                                        self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
        x_sub1, _ = self.st_gcn_pool[1](x_sub1, self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
        x_sub1 = x_sub1.view(N, -1, T, len(self.graph.part))

        x_pool_1 = self.graph_max_pool(x_sub1, (1, len(self.graph.part)))  # N, 512, T, 1
        x_pool_1 = self.conv4(x_pool_1)  # N, C, T, 1

        x_up_sub = torch.cat((x_pool_1.repeat(1, 1, 1, len(self.graph.part)), x_sub1), 1)  # N, 1024, T, 5
        x_up_sub = self.conv2(x_up_sub)  # N, C, T, 5

        # upsample
        #x_up = torch.zeros((N * M, fc_out, T, V)).cuda()
        #x_up = torch.zeros((N * M, fc_out, T, V)).cpu() # wally


        for i in range(len(self.graph.part)):
            num_node = len(self.graph.part[i])
            x_wally = x_up_sub[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, num_node)
            x_wally_sub1 = torch.cat((x_wally_sub1, x_wally), -1) if i > 0 else x_wally

        #x_up_wally = torch.zeros((N * M, fc_out, T, V)).cpu()
        qwer_a = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12], [13, 14, 15, 16]]
        #qwer_b = [[11, 12, 13], [14, 15, 16], [4, 5, 6], [1, 2, 3], [0], [7, 8, 9, 10]]
        tmeptemp_0 = x_wally_sub1[:, :, :, qwer_a[0]]  # [11, 12, 13]
        tmeptemp_1 = x_wally_sub1[:, :, :, qwer_a[1]]  # [14, 15, 16]
        tmeptemp_2 = x_wally_sub1[:, :, :, qwer_a[2]]  # [4, 5, 6]
        tmeptemp_3 = x_wally_sub1[:, :, :, qwer_a[3]]  # [1, 2, 3]
        tmeptemp_4 = x_wally_sub1[:, :, :, qwer_a[4]]  # [0]
        tmeptemp_5 = x_wally_sub1[:, :, :, qwer_a[5]]  # [7, 8, 9, 10]

        tempWally_0 = torch.cat((tmeptemp_4, tmeptemp_3), -1)
        tempWally_0 = torch.cat((tempWally_0, tmeptemp_2), -1)
        tempWally_0 = torch.cat((tempWally_0, tmeptemp_5), -1)
        tempWally_0 = torch.cat((tempWally_0, tmeptemp_0), -1)
        tempWally_0 = torch.cat((tempWally_0, tmeptemp_1), -1)

        '''
        othertempWally_0 = torch.cat((x_up_sub, x_up_sub), -1)
        othertempWally_0 = torch.cat((othertempWally_0, x_up_sub), -1)
        othertempWally_0 = torch.cat((othertempWally_0, x_up_sub[:, :, :, 0].unsqueeze(-1).repeat(1, 1, 1, 2)), -1)
        #otertempWally_0 = torch.cat((othertempWally_0, x_up_sub[:, :, :, 0].unsqueeze(-1).repeat(1, 1, 1, 1), -1))
        '''

        #for i in range(len(self.graph.part)):
        #    num_node = len(self.graph.part[i])
        #    x_up_wally[:, :, :, self.graph.part[i]] = x_wally_sub1[:, :, :, qwer_a[i]]

        #for i in range(len(self.graph.part)):
        #    num_node = len(self.graph.part[i])
        #    x_up[:, :, :, self.graph.part[i]] = x_up_sub[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, num_node)  # wally

        #wally_0 = torch.eq(tempWally_0, x_up)
        #wally_1 = wally_0.float()
        #wally_2 = torch.sum(wally_1)
        #wally_3 = torch.numel(wally_1)



        # for non-local and fcn
        #x = torch.cat((x, x_up), 1) #wally
        x = torch.cat((x, tempWally_0), 1) #wally
        #x = torch.cat((x, tempWally_0), 1)
        x = self.non_local(x)  # N, 2C, T, V
        x = self.fcn(x)  # N, 3, T, V

        # output
        x = x.view(N, M, -1, T, V).permute(0, 2, 3, 4, 1).contiguous()  # N, C, T, V, M
        if out_all_frame:
            x_out = x
        else:
            x_out = x[:, :, self.pad].unsqueeze(2)
        return x_out


class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size :number of the node clusters

        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, 1, T*V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, T*V, T*V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, 1, T*V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, T*V, T*V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the kernel size
            :math:`T` is a length of sequence,
            :math:`V` is the number of graph nodes of each frame.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.05,
                 residual=True):

        super().__init__()
        self.inplace = True

        self.momentum = 0.1
        self.gcn = st_gcn_ConvTemporalGraphical(in_channels, out_channels, kernel_size)

        self.tcn = nn.Sequential(

            nn.BatchNorm2d(out_channels, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(dropout),
            nn.Conv2d(
                out_channels,
                out_channels,
                (1, 1),
                (stride, 1),
                padding=0,
            ),
            nn.BatchNorm2d(out_channels, momentum=self.momentum),
            nn.Dropout(dropout, inplace=self.inplace),

        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels, momentum=self.momentum),
            )

        self.relu = nn.ReLU(inplace=self.inplace)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res

        return self.relu(x), A


class WrapSTGCN(nn.Module):
    def __init__(self, p_dropout):
        super(WrapSTGCN, self).__init__()
        if p_dropout < 0:
            print('use default stgcn')
            self.stgcn = Model_defaultDropout()
        else:
            self.stgcn = Model(dropout=p_dropout)

    def forward(self, x):
        """
        input: bx16x2 / bx32
        output: bx16x3
        """
        if len(x.shape) == 2:
            x = x.view(x.shape[0], 16, 2)
        # add one joint: 16 to 17
        Ct = torch.Tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]).transpose(1, 0)

        Ct = Ct.to(x.device)
        C = Ct.repeat([x.size(0), 1, 1]).view(-1, 16, 17)
        x = x.view(x.size(0), -1, 2)  # nx16x2
        x = x.permute(0, 2, 1).contiguous()  # nx2x16
        x = torch.matmul(x, C)  # nx2x17

        # process to stgcn
        x = x.unsqueeze(2).unsqueeze(-1)  # nx2x1x17x1
        out = self.stgcn(x)  # nx3x1x17x1
        #out2 = out
        #out_alt = out.squeeze() # nx3x17
        #out = out.squeeze(dim=0)  # wally
        #out = out.squeeze(dim=1)  # wally
        #out = out.squeeze(dim=2)  # wally
        #out = out.resize_((3, 17))
        out = out.view((3, 17))

        # remove the joint: 17 to 16
        Ct17 = torch.Tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]).transpose(1, 0)
        Ct17 = Ct17.to(out.device)
        C17 = Ct17.repeat([out.size(0), 1, 1]).view(-1, 17, 16)
        out = torch.matmul(out, C17)  # nx2x17

        out = out.permute(0, 2, 1).contiguous()  # nx16x3

        #////////////////
        #out_wally_here_0 = out[0]
        #out_wally_here_1 = out[1]
        #wally_0 = torch.eq(out_wally_here_0, out_wally_here_1)
        #wally_1 = wally_0.float()
        #wally_2 = torch.sum(wally_1)
        #wally_3 = torch.numel(wally_1)
        #////////////////


        return out[0]


# test code here.
if __name__ == '__main__':
    #input = torch.randn((12, 16, 2), requires_grad=True).cuda()
    #target = torch.randn((12, 16, 3), requires_grad=True).cuda()
    #model = WrapSTGCN().cuda()
    input = torch.randn((12, 16, 2), requires_grad=True).cpu() #wally
    target = torch.randn((12, 16, 3), requires_grad=True).cpu() #wally
    model = WrapSTGCN().cpu() #wally
    loss = nn.MSELoss()
    pre = model(input)
    l = loss(pre, target)
    l.backward()

    print('ss')
