import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 유틸리티 함수 ---

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    'linear' 또는 'cosine' 스케줄에 따라 beta 값을 생성합니다.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        s = 0.008
        t = np.linspace(0, num_diffusion_timesteps, num_diffusion_timesteps + 1)
        t = t / num_diffusion_timesteps
        alphas_cumprod = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

# --- 핵심 확산 모델 클래스 ---

class TimeSeriesDiffusion:
    """
    시계열 데이터 생성을 위한 가우시안 확산 모델 클래스.
    OpenAI의 코드를 기반으로 시계열 문제에 맞게 재구성 및 간소화되었습니다.

    :param model: 노이즈 예측을 수행할 신경망 모델.
    :param betas: 1-D numpy 배열 형태의 노이즈 스케줄.
    :param past_len: 조건으로 사용될 과거 데이터의 길이.
    :param future_len: 예측/생성할 미래 데이터의 길이.
    :param device: 연산에 사용할 디바이스 ('cuda' or 'cpu').
    """
    def __init__(self, model, betas, past_len, future_len, device):
        self.model = model
        self.past_len = past_len
        self.future_len = future_len
        self.device = device

        # 1. 모델의 예측 목표를 '노이즈(epsilon)'로 고정
        # 2. Loss 함수를 'MSE'로 고정
        # 3. Reverse Process의 분산(variance)을 고정된 값(posterior_variance)으로 사용

        # float64 타입으로 정확도 유지
        betas = torch.tensor(betas, dtype=torch.float64).to(device)
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # --- Forward Process q(x_t | x_0)에 필요한 계수 ---
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # --- Reverse Process q(x_{t-1} | x_t, x_0)에 필요한 계수 ---
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, arr, t, broadcast_shape):
        """1차원 텐서 arr에서 t 인덱스에 해당하는 값을 추출하고 broadcast_shape 형태로 변형합니다."""
        res = arr.to(self.device)[t].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """예측된 노이즈(eps)를 사용하여 x_0를 예측합니다."""
        return (
            self._extract(1.0 / self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod, t, x_t.shape) * eps
        )

    def _q_posterior_mean_variance(self, x_start, x_t, t):
        """q(x_{t-1} | x_t, x_0)의 평균과 분산을 계산합니다."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_start, t, noise=None):
        """Forward Process: 원본 데이터 x_start에 t 시점만큼의 노이즈를 주입합니다."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_mean_variance(self, x_t, t, cond, clip_denoised=True):
        """Reverse Process의 한 스텝: 모델을 이용해 평균과 분산을 계산합니다."""
        # 1. 모델을 통해 노이즈 예측
        pred_noise = self.model(x_t, t, cond=cond)
        
        # 2. 예측된 노이즈를 통해 x_0 예측
        pred_xstart = self._predict_xstart_from_eps(x_t, t, pred_noise)
        if clip_denoised:
            pred_xstart = torch.clamp(pred_xstart, -1.0, 1.0) # 데이터 정규화 범위에 따라 조절

        # 3. 예측된 x_0를 이용해 posterior mean, variance 계산
        model_mean, model_variance, model_log_variance = self._q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    @torch.no_grad()
    def p_sample(self, x_t, t, cond):
        """한 타임스텝의 샘플링(x_{t-1} 생성)을 수행합니다."""
        out = self.p_mean_variance(x_t, t, cond)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, progress=False):
        """전체 타임스텝에 걸쳐 샘플링을 진행하여 최종 결과물을 생성합니다."""
        self.model.eval()
        img = torch.randn(shape, device=self.device)
        
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
            
        for i in indices:
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            out = self.p_sample(img, t, cond)
            img = out["sample"]
        
        self.model.train()
        return img

    def training_losses(self, x_start, t, cond, noise=None):
        """학습을 위한 MSE 손실을 계산합니다."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise=noise)
        predicted_noise = self.model(x_t, t, cond=cond)
        
        # 실제 노이즈와 예측된 노이즈 간의 MSE Loss 계산
        loss = F.mse_loss(noise, predicted_noise)
        return {"loss": loss}