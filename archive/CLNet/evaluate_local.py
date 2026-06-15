import os
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np

from models.clnet import CLNet
from utils.statics import evaluator


def evaluate_clnet(data_dir='../data', scenario='in', cr=32,
                   encoder_path='./Modelsave/32encoder.pth.tar',
                   decoder_path='./Modelsave/32decoder.pth.tar',
                   batch_size=200):
    device = torch.device('cpu')

    # Load model
    model = CLNet(reduction=cr)
    enc_ckpt = torch.load(encoder_path, map_location='cpu')
    dec_ckpt = torch.load(decoder_path, map_location='cpu')
    model.encoder.load_state_dict(enc_ckpt['state_dict'], strict=False)
    model.decoder.load_state_dict(dec_ckpt['state_dict'], strict=False)
    model.to(device)
    model.eval()

    print(f"Loaded encoder from {encoder_path}")
    print(f"Loaded decoder from {decoder_path}")

    # Load test data
    dir_test = os.path.join(data_dir, f"DATA_Htest{scenario}.mat")
    dir_raw = os.path.join(data_dir, f"DATA_HtestF{scenario}_all.mat")

    data_test = sio.loadmat(dir_test)['HT']
    data_test = torch.tensor(data_test, dtype=torch.float32).view(data_test.shape[0], 2, 32, 32)

    raw_test = sio.loadmat(dir_raw)['HF_all']
    real = torch.tensor(np.real(raw_test), dtype=torch.float32)
    imag = torch.tensor(np.imag(raw_test), dtype=torch.float32)
    raw_test = torch.cat((real.view(raw_test.shape[0], 32, 125, 1),
                          imag.view(raw_test.shape[0], 32, 125, 1)), dim=3)

    n_samples = data_test.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    criterion = nn.MSELoss()
    total_loss = 0.0
    total_rho = 0.0
    total_nmse = 0.0

    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            sparse_gt = data_test[start:end].to(device)
            raw_gt = raw_test[start:end].to(device)

            sparse_pred = model(sparse_gt)
            loss = criterion(sparse_pred, sparse_gt)
            rho, nmse = evaluator(sparse_pred, sparse_gt, raw_gt)

            bs = end - start
            total_loss += loss.item() * bs
            total_rho += rho.item() * bs
            total_nmse += nmse.item() * bs

    avg_loss = total_loss / n_samples
    avg_rho = total_rho / n_samples
    avg_nmse = total_nmse / n_samples

    print(f"\nCLNet evaluation results (scenario={scenario}, cr=1/{cr}):")
    print(f"  Loss: {avg_loss:.3e}")
    print(f"  rho:  {avg_rho:.3e}")
    print(f"  NMSE: {avg_nmse:.3e} dB")

    return avg_loss, avg_rho, avg_nmse


if __name__ == '__main__':
    # Local checkpoint '32encoder/decoder' actually has encoded_dim=32, i.e. reduction=64
    evaluate_clnet(cr=64, encoder_path='./Modelsave/32encoder.pth.tar', decoder_path='./Modelsave/32decoder.pth.tar')
