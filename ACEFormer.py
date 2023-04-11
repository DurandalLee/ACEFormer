import time
import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as Data
import numpy as np
import pandas as pd
import os
import sys
import multiprocessing
from moduledata import EmdData
from attnset import ProbabilityAttention, FullAttention
from pretreatment import PositionalEmbedding, ExpandEmbedding
from module import Distilling, CrossLayer



class AllData:
    def __init__(self, source_data: pd.DataFrame, verify_size: int, test_size: int, unit_size: int, predict_size: int, emd_col: list, result_col: list, back_num: int, data_type, emd_type: int = 3):
        super(AllData, self).__init__()
        dataframe = source_data
        self.true_train_set,self.true_verify_set,self.true_test_set = [], [], []
        self.former_train_set,self.former_verify_set,self.former_test_set = [], [], []

        split_index = - back_num * test_size - verify_size - 1
        train_tmp = dataframe.iloc[:split_index].reset_index(drop=True)
        self.true_train_set.append(train_tmp)
        self.former_train_set.append(data_type(train_tmp, unit_size, predict_size, emd_col=emd_col, result_col=result_col, emd_type=emd_type))

        for _ in range(back_num):
            # train data
            train_tmp = dataframe.iloc[split_index-unit_size+predict_size: split_index+test_size if split_index<-test_size else -1].reset_index(drop=True)
            self.true_train_set.append(train_tmp)
            self.former_train_set.append(data_type(train_tmp, unit_size, predict_size, emd_col=emd_col, result_col=result_col, emd_type=emd_type))

            # verify data
            split_index += verify_size
            verify_tmp = dataframe.iloc[split_index-verify_size-unit_size+predict_size: split_index].reset_index(drop=True)
            self.true_verify_set.append(verify_tmp)
            self.former_verify_set.append(data_type(verify_tmp, unit_size, predict_size, emd_col=emd_col, result_col=result_col, emd_type=emd_type))

            # test data
            split_index += test_size
            test_tmp = dataframe.iloc[split_index-test_size-unit_size+predict_size: split_index].reset_index(drop=True)
            self.true_test_set.append(test_tmp)
            self.former_test_set.append(data_type(test_tmp, unit_size, predict_size, emd_col=emd_col, result_col=result_col, emd_type=emd_type))
            
            # index
            split_index -= verify_size

    def get_data(self):
        return self.former_train_set, self.former_verify_set, self.former_test_set

    def get_not_normaliza_data(self):
        return self.true_train_set, self.true_verify_set, self.true_test_set

class ACEFormer(nn.Module):
    def __init__(self, data_dim: int, embed_dim: int, forward_dim: int, unit_size: int, dis_layer: int = 3, attn_layer: int = 2, factor: int = 5, dropout: float = 0.1, activation: str = "relu"):
        super(ACEFormer, self).__init__()
        self.dis_layer = dis_layer
        self.attn_layer = attn_layer

        # pretreatment module
        ## data embedding
        self.ExpandConv = ExpandEmbedding(data_dim, embed_dim)
        ## local position
        self.position_emb = PositionalEmbedding(embed_dim)
        ## dropout
        self.dropout = nn.Dropout(p=dropout)

        # distillation module
        ## temporal perception mechanism
        self.temporal = nn.ModuleList()
        ## distillation mechanism
        self.dis_attn = nn.ModuleList()
        self.distill = nn.ModuleList()
        ## create distillation module
        for num in range(dis_layer):
            embed_tmp = embed_dim // pow(2, num)
            self.dis_attn.append(
                CrossLayer(
                    ProbabilityAttention(embed_tmp, n_heads=8, factor=factor),
                    embed_dim=embed_tmp, forward_dim=forward_dim, 
                    dropout=dropout, activation=activation
                )
            )
            self.distill.append(Distilling(embed_tmp))
            self.hidden_local.append(nn.Linear(unit_size, embed_tmp // 2))

        # attention module
        self.attn = nn.ModuleList(
            CrossLayer(
                FullAttention(embed_tmp // 2, n_heads=8, factor=factor),
                embed_dim=embed_tmp // 2, forward_dim=embed_tmp * 2,
                dropout=dropout, activation=activation
            ) for _ in range(attn_layer)
        )

        # projection
        self.full_connect = nn.Linear(embed_tmp // 2, 1, bias=True)

    def forward(self, data: torch.tensor):
        # data embedding
        data_emb = self.ExpandConv(data)
        local_position = self.position_emb(data)
        dis_input = data_emb + local_position
        dis_output = self.dropout(dis_input)

        # distilling module
        for i in range(self.dis_layer):
            attn_res = self.dis_attn[i](dis_output, dis_output)
            dis_res = self.distill[i](attn_res)
            hid_res = self.temporal[i](dis_output)
            dis_output = dis_res + hid_res
        
        # attention dealt
        attn_output = dis_output
        for layer in self.attn:
            attn_output = layer(attn_output, dis_output)

        # projection
        output = self.full_connect(attn_output)

        return output

def train(model, train_data: Data.Dataset, batch_size: int, device: str = "cpu", iteration: int = 2000):
    # data to DataLoader
    train_loader = Data.DataLoader(train_data, batch_size)
    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = opt.Adam(model.parameters(), lr=0.001)

    # trainning model
    model.train()
    start = time.time()
    for epoch in range(iteration):
        ## calculate the loss
        batch_count, loss_count = 0, 0.0

        for data, _, true_data in train_loader:
            data = data.float().to(device)
            true_data = true_data.float().to(device)

            outputs = model(data)
            loss = criterion(outputs, true_data)
            loss_count += loss.cpu().data
            batch_count += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            end = time.time()
            print("epoch=" + str(epoch) + ", loss=" + str(loss_count/batch_count) + ", use time=" + str(int(end - start)) + "s, in " + time.strftime("%m-%d %H:%M:%S", time.localtime()) + ", predict next epoch in " + time.strftime("%m-%d %H:%M:%S", time.localtime(time.time() + end - start)))
            start = time.time()

    return model

def test(model, test_data: EmdData, predict_size: int, device: str):
    model.eval()
    true, predict = [], []
    with torch.no_grad():
        for (data, stamp, true_data) in test_data:
            data = torch.tensor(data).unsqueeze(0).float().to(device)
            stamp = torch.tensor(stamp).unsqueeze(0).int().to(device)

            true.append(true_data[-predict_size])
            outputs, _, _ = model(data, stamp)
            predict.append(outputs.reshape(-1)[-predict_size:].tolist())

    true, predict = test_data.anti_normalize_data(np.array(true), np.array(predict))
    return true, predict

def run_model(source_data: pd.DataFrame, index: int, device: str, backtest_num: int, iteration: int, save_path: str):
    print("Start experiment index : " + str(index) + " ......")
    # hyper parameter
    batch_size = 64
    emd_col = ['close', 'close_x', 'close_y', 'vol', 'vol_x', 'vol_y']
    result_col = ['close']

    start_time = time.time()
    # product dataset for model
    data_set = AllData(source_data=source_data, verify_size=50, test_size=100, unit_size=30, predict_size=5, emd_col=emd_col, result_col=result_col, back_num=backtest_num, data_type=EmdData)
    former_train_set, former_verify_set, former_test_set = data_set.get_data()
    true_train_set, true_verify_set, true_test_set = data_set.get_not_normaliza_data()

    # create model
    print("create model")
    model = ACEFormer(data_dim=len(emd_col), embed_dim=64, forward_dim=256, unit_size=30, dis_layer=3, attn_layer=2, dropout=0.1, factor=5).to(device)

    train_true_set, train_predict_set = [], []
    verify_true_set, verify_predict_set = [], []
    test_true_set, test_predict_set = [], []
    for i in range(backtest_num):
        # training model with train set
        model = train(model, former_train_set[i], batch_size, device, iteration)
        # training
        true, predict = test(model, former_train_set[i], predict_size=5, device=dev)
        train_true_set.append(true)
        train_predict_set.append(predict)
        # verify
        true, predict = test(model, former_verify_set[i], predict_size=5, device=dev)
        verify_true_set.append(true)
        verify_predict_set.append(predict)
        # test
        true, predict = test(model, former_test_set[i], predict_size=5, device=dev)
        test_true_set.append(true)
        test_predict_set.append(predict)

    all_set_dict = {
        "true_train_set": true_train_set,
        "true_verify_set": true_verify_set,
        "true_test_set": true_test_set,
        "train_true_set": train_true_set,
        "train_predict_set": train_predict_set,
        "verify_true_set": verify_true_set,
        "verify_predict_set": verify_predict_set,
        "test_true_set": test_true_set,
        "test_predict_set": test_predict_set
    }

    np.save(save_path.format(index), all_set_dict)
    
    print("Save result to " + save_path.format(index) + ", spend time " + str((time.time() - start_time) // 60 // 60) + "h " + str((time.time() - start_time) // 60 % 60) + "min.")

if __name__ == "__main__":
    print('args : ', sys.argv)
    print('script name : ', sys.argv[0])
    # process and GPU use
    dev = sys.argv[1]
    # experiment times for each data
    model_time = int(sys.argv[2])
    # path for save predict result
    result_save = sys.argv[3]
    # data file
    data_path = sys.argv[4]
    # the number of the dataset splits
    back_num = int(sys.argv[5])
    # iteration number
    itera_num = int(sys.argv[6])
    # path of save result
    result_path = result_save + "result_set_" + data_path[10:-4] + "_{}.npy"

    # multiplt processing
    pool = multiprocessing.Pool(model_time)
    source_data = pd.read_csv(data_path)
    for model_i in range(1, model_time + 1):
        pool.apply_async(run_model, (source_data, model_i, dev, back_num, itera_num, result_path))

    pool.close()
    pool.join()
    
    print("Finish training all experment models.")