import numpy as np
import torch as th
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data.data_factory import data_provider
from guided_diffusion.script_util import create_gaussian_diffusion
from model.mlp import MLP
from model.lstm import LSTMModel
from model.dlinear import DLinear
from model.timesnet import TimesNet
from model.ns_transformer import NsTransformer
from src.hyperdm import HyperDM
from src.util import parse_train_args, metric, normalize_range

if __name__ == "__main__":
    args = parse_train_args()

    # args setting
    args.data = 'DOW'
    args.root_path = './dataset/'
    data_type = 'return' if args.root_path=='./dataset/' else 'price'
    args.freq = 'd'
    args.seed = 1

    args.seq_len = 63
    args.label_len = 0
    args.pred_len = 5
    args.inverse = True 
    args.train_end = '2018-12-31' # '2017-12-31' for DOW, '2016-12-31' for SNP
    args.valid_end = '2020-12-31' # '2019-06-30' for DOW, '2019-06-30' for SNP

    args.pretrain = True
    args.model = 'lstm'  # 'mlp', 'lstm', 'dlinear', 'timesnet', 'ns_transformer'

    if args.data == 'DOW':
        args.enc_in = 18
        args.dec_in = 18
        args.c_out = 18
    elif args.data == 'SNP':
        args.enc_in = 30
        args.dec_in = 30
        args.c_out = 30

    if args.model == 'timesnet':
        args.d_model = 64
        args.d_ff = 64
        args.top_k = 5
        args.num_kernels = 6
    else:
        args.d_model = 512
        args.d_ff = 2048

    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    
    args.factor = 3
    args.moving_avg = 25
    args.embed = 'timeF'
    args.dropout = 0.2
    args.activation = 'gelu'

    # ns-transformer specific args
    args.p_hidden_dims = [64, 64]
    args.p_hidden_layers = 2

    args.is_training = True
    args.num_epochs = 10
    args.lr = 1e-4
    args.batch_size = 32
    args.patience = 5

    args.M = 10
    args.N = 100
    args.diffusion_steps = 1000
    args.hyper_net_dims = [1, 24, 24, 24, 24, 24]

    # print args
    print(f"Training: {args.is_training}, Data: {args.data}, Model: {args.model}, Seq Len: {args.seq_len}, Pred Len: {args.pred_len}, Seed: {args.seed}")

    # Seed for reproducible results.
    if not args.seed is None:
        rng = th.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = "cuda" if th.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 결과 저장 경로 생성
    result_dir = os.path.join(
        "results", 
        "hyperdm",
        args.data, 
        f"{args.model}_{args.seq_len}_{args.pred_len}_{args.seed}"
    )
    os.makedirs(result_dir, exist_ok=True)

    # Pretrain directory
    pretrain_dir = os.path.join(
        "results", 
        "pretrain",
        args.data, 
        f"{args.model}_{args.seq_len}_{args.pred_len}_{args.seed}"
    )

    if args.is_training: # Training mode
        train_data, train_loader = data_provider(args, flag='train')
        val_data, val_loader = data_provider(args, flag='val')
        test_data, test_loader = data_provider(args, flag='test')

        # 모델 생성
        if args.model == 'ns_transformer':
            ts_model = NsTransformer(args).to(device)
        elif args.model == 'dlinear':
            ts_model = DLinear(args).to(device) 
        elif args.model == 'timesnet':
            ts_model = TimesNet(args).to(device)
        elif args.model == 'mlp':
            ts_model = MLP(args).to(device)
        elif args.model == 'lstm':
            ts_model = LSTMModel(args).to(device)
        else:
            raise ValueError(f"Model {args.model} is not supported.")
        
        if args.pretrain:
            if os.path.exists(os.path.join(pretrain_dir, 'best_model.pth')):
                print(f"Loading pre-trained model from {pretrain_dir}")
                ts_model.load_state_dict(torch.load(os.path.join(pretrain_dir, 'best_model.pth')))
        else:
            print("No pre-trained model found. Training from scratch.")

        diffusion = create_gaussian_diffusion(
            steps=args.diffusion_steps,
            predict_xstart=True
        )
        hyperdm = HyperDM(ts_model, args.hyper_net_dims, diffusion).to(device)
        hyperdm.print_stats()
        hyperdm.train()

        optimizer = torch.optim.AdamW(hyperdm.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=3)

        best_val_loss = float('inf')
        patience = 0

        for epoch in range(args.num_epochs):
            hyperdm.train()
            train_losses = []
            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
            
                
                # batch_x_mark = batch_x_mark.float().to(device)
                # batch_y_mark = batch_y_mark.float().to(device)
                t = th.randint(0, args.diffusion_steps, (len(batch_x), ), device=device)

                net = hyperdm.sample_network(device)

                optimizer.zero_grad()
                # decoder input 생성
                # dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                # outputs = ts_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # f_dim = 0
                # outputs = outputs[:, -args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                # loss = criterion(outputs, batch_y)

                print("--- Shape Debug ---")
                print(f"batch_x shape:      {batch_x.shape}")
                print(f"batch_y shape:      {batch_y.shape}")
                print(f"batch_x_mark shape: {batch_x_mark.shape}")
                print(f"batch_y_mark shape: {batch_y_mark.shape}")
                print("--------------------")
                
                
                loss = hyperdm.diffusion.training_losses(net, batch_x, t, model_kwargs={"y": batch_y})["loss"].mean()
                train_losses.append(loss.item())

                loss.backward()
                optimizer.step()

            # 검증
            ts_model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    # batch_x_mark = batch_x_mark.float().to(device)
                    # batch_y_mark = batch_y_mark.float().to(device)

                    # dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    # dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                    # outputs = ts_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # f_dim = 0
                    # outputs = outputs[:, -args.pred_len:, f_dim:]
                    # batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                    # loss = criterion(outputs, batch_y)

                    net = hyperdm.sample_network(device)
                    loss = hyperdm.diffusion.training_losses(net, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    val_losses.append(loss.item())

            avg_train_loss = np.average(train_losses)
            avg_val_loss = np.average(val_losses)
            print(f"Epoch {epoch+1}/{args.num_epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

            scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience = 0
                torch.save(ts_model.state_dict(), os.path.join(result_dir, 'best_model.pth'))
            else:
                patience += 1
                if patience >= args.patience:
                    print("Early stopping!")
                    break

        # 테스트 -> 이렇게 바로 inference로 이어져도 ts_model 파라미터는 그대로인지 확인해야함

        diffusion = create_gaussian_diffusion(steps=args.diffusion_steps,
                                          predict_xstart=True,
                                          timestep_respacing="ddim25")
        hyperdm = HyperDM(ts_model, args.hyper_net_dims, diffusion).to(device)
        hyperdm.load_state_dict(th.load(args.checkpoint, weights_only=True))
        hyperdm.eval()

        preds, trues, eus, aus = [], [], [], []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                # batch_x_mark = batch_x_mark.float().to(device)
                # batch_y_mark = batch_y_mark.float().to(device)

                # dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                # outputs = ts_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # f_dim = 0
                # outputs = outputs[:, -args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

                batch_mean, batch_var = hyperdm.get_mean_variance(M=args.M,
                                                        N=args.N,
                                                        condition=batch_x,
                                                        device=device,
                                                        progress=True)
                
                pred = batch_mean.mean(dim=0).cpu().numpy()
                eu = batch_mean.var(dim=0).cpu().numpy()
                au = batch_var.mean(dim=0).cpu().numpy()

                preds.append(pred)
                trues.append(batch_y.detach().cpu().numpy())
                eus.append(eu)
                aus.append(au)
                
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        eus = np.concatenate(eus, axis=0)
        aus = np.concatenate(aus, axis=0)

        # Normalize uncertainty for visualization purposes -> 이건 고민
        # eu_norm = normalize_range(eus, low=0, high=1)
        # au_norm = normalize_range(aus, low=0, high=1)

        print(f"Preds shape: {preds.shape}, Trues shape: {trues.shape}, EUs shape: {eus.shape}, AUs shape: {aus.shape}")

        mae, mse = metric(preds, trues)
        print(f"Test MSE: {mse}, Test MAE: {mae}")

        if test_data.scale and args.inverse:
            shape = preds.shape
            preds_inverse = test_data.inverse_transform(preds.reshape(-1, shape[-1])).reshape(shape)
            trues_inverse = test_data.inverse_transform(trues.reshape(-1, shape[-1])).reshape(shape)
        else:
            preds_inverse = preds
            trues_inverse = trues

        # print(f"Preds shape: {preds.shape}, Trues shape: {trues.shape}")
        mae, mse = metric(preds_inverse, trues_inverse)
        print(f"Inverse Test MSE: {mse}, Inverse Test MAE: {mae}")

        # === UP/DOWN 방향성 예측 정확도 계산 ===
        # preds_inverse, trues_inverse shape: (batch, pred_len, feature)
        pred_updown = (preds_inverse > 0).astype(int)
        target_updown = (trues_inverse > 0).astype(int)
        updown_acc = (pred_updown == target_updown).mean()
        print(f"Test Up/Down Accuracy: {updown_acc * 100:.2f}%")

        np.save(os.path.join(result_dir, 'preds.npy'), preds)
        np.save(os.path.join(result_dir, 'trues.npy'), trues)


    else: # Inference mode
        test_data, test_loader = data_provider(args, flag='test')
        # # 모델 생성
        # if args.model == 'ns_transformer':
        #     ts_model = NsTransformer(args).to(device)
        # elif args.model == 'dlinear':
        #     ts_model = DLinear(args).to(device) 
        # elif args.model == 'timesnet':
        #     ts_model = TimesNet(args).to(device)
        # else:
        #     raise ValueError(f"Model {args.model} is not supported.")
        # ts_model.load_state_dict(torch.load(os.path.join(result_dir, 'best_model.pth')))
        # ts_model.eval()
        # preds, trues = [], []
        # with torch.no_grad():
        #     for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        #         batch_x = batch_x.float().to(device)
        #         batch_y = batch_y.float().to(device)
        #         batch_x_mark = batch_x_mark.float().to(device)
        #         batch_y_mark = batch_y_mark.float().to(device)
        #         dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        #         dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
        #         outputs = ts_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        #         f_dim = 0
        #         outputs = outputs[:, -args.pred_len:, f_dim:]
        #         batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

        #         preds.append(outputs.detach().cpu().numpy())
        #         trues.append(batch_y.detach().cpu().numpy())
                
        # preds = np.concatenate(preds, axis=0)
        # trues = np.concatenate(trues, axis=0)

        # mae, mse = metric(preds, trues)
        # print(f"Test MSE: {mse}, Test MAE: {mae}")

        # if test_data.scale and args.inverse:
        #     shape = preds.shape
        #     preds_inverse = test_data.inverse_transform(preds.reshape(-1, shape[-1])).reshape(shape)
        #     trues_inverse = test_data.inverse_transform(trues.reshape(-1, shape[-1])).reshape(shape)
        # else:
        #     preds_inverse = preds
        #     trues_inverse = trues

        # # print(f"Preds shape: {preds.shape}, Trues shape: {trues.shape}")
        # mae, mse = metric(preds_inverse, trues_inverse)
        # print(f"Inverse Test MSE: {mse}, Inverse Test MAE: {mae}")

        # # === UP/DOWN 방향성 예측 정확도 계산 ===
        # # preds_inverse, trues_inverse shape: (batch, pred_len, feature)
        # pred_updown = (preds_inverse > 0).astype(int)
        # target_updown = (trues_inverse > 0).astype(int)
        # updown_acc = (pred_updown == target_updown).mean()
        # print(f"Test Up/Down Accuracy: {updown_acc * 100:.2f}%")

        # np.save(os.path.join(result_dir, 'preds.npy'), preds)
        # np.save(os.path.join(result_dir, 'trues.npy'), trues)
