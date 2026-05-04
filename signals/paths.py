"""Raw probabilistic-path inference for Kronos.

`KronosPredictor.predict` averages all sample paths into a single mean forecast.
For decision support we need the per-sample paths so we can compute
percentiles, dispersion, and touch probabilities. This module runs the same
auto-regressive loop but returns every sampled path.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps, sample_from_logits


PRICE_COLS = ["open", "high", "low", "close"]
VOL_COL = "volume"
AMT_COL = "amount"


class PathForecaster:
    """Run Kronos and return per-sample forecast paths (no averaging)."""

    def __init__(
        self,
        model: Kronos,
        tokenizer: KronosTokenizer,
        device: Optional[str] = None,
        max_context: int = 512,
        clip: float = 5.0,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.max_context = max_context
        self.clip = clip
        self.model = model.to(device)
        self.tokenizer = tokenizer.to(device)

    def forecast_paths(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 100,
        verbose: bool = False,
    ) -> np.ndarray:
        """Return forecast paths of shape (sample_count, pred_len, 6).

        Columns are [open, high, low, close, volume, amount] in the same units
        as the input DataFrame.
        """
        df = df.copy()
        if VOL_COL not in df.columns:
            df[VOL_COL] = 0.0
            df[AMT_COL] = 0.0
        if AMT_COL not in df.columns:
            df[AMT_COL] = df[VOL_COL] * df[PRICE_COLS].mean(axis=1)

        if df[PRICE_COLS + [VOL_COL, AMT_COL]].isnull().values.any():
            raise ValueError("Input DataFrame has NaN values in price/volume columns.")

        x = df[PRICE_COLS + [VOL_COL, AMT_COL]].values.astype(np.float32)
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -self.clip, self.clip)

        x_stamp = calc_time_stamps(x_timestamp).values.astype(np.float32)
        y_stamp = calc_time_stamps(y_timestamp).values.astype(np.float32)

        x_t = torch.from_numpy(x_norm[None, :]).to(self.device)
        x_stamp_t = torch.from_numpy(x_stamp[None, :]).to(self.device)
        y_stamp_t = torch.from_numpy(y_stamp[None, :]).to(self.device)

        paths = _inference_paths(
            self.tokenizer,
            self.model,
            x_t,
            x_stamp_t,
            y_stamp_t,
            self.max_context,
            pred_len,
            self.clip,
            T,
            top_k,
            top_p,
            sample_count,
            verbose,
        )
        # paths: (1, sample_count, decoded_len, 6) — slice forecast portion + denormalize
        paths = paths[0, :, -pred_len:, :]
        return paths * (x_std + 1e-5) + x_mean


def _inference_paths(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip,
    T,
    top_k,
    top_p,
    sample_count,
    verbose,
):
    """Adapted from `model.kronos.auto_regressive_inference` but returns
    per-sample paths instead of the cross-sample mean."""
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device

        x = (
            x.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x.size(1), x.size(2))
            .to(device)
        )
        x_stamp = (
            x_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x_stamp.size(1), x_stamp.size(2))
            .to(device)
        )
        y_stamp = (
            y_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, y_stamp.size(1), y_stamp.size(2))
            .to(device)
        )

        x_token = tokenizer.encode(x, half=True)

        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx : start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx : start_idx + buffer_len]

        ran = trange if verbose else range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(
                s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(
                s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)
        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        return z.cpu().numpy()
