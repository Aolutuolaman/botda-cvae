import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
import torch.fft
import pandas as pd
from tqdm import tqdm
# ====================== 配置 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M_FACTOR = 3
MAX_LEN = 1024
LATENT_DIM = 128
HIDDEN_DIM = 1024
COND_DIM = 3
MAX_FE = 700
MAX_FADING = 0.5
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_OUTPUT_BASE = os.path.join(PROJECT_ROOT, "outputs")
MODEL_NAME = "universal_cvae_v2_best.pt"
# ====================== 工具函数 ======================
def make_condition(fe: int, target_len: int, fading: float) -> torch.Tensor:
    fe_norm = fe / MAX_FE
    len_norm = target_len / MAX_LEN
    fading_norm = fading / MAX_FADING
    return torch.FloatTensor([fe_norm, len_norm, fading_norm])
def apply_edfa_envelope(seq_binary: torch.Tensor, fading: float) -> torch.Tensor:
    if fading <= 0:
        return seq_binary.float()
    B, L = seq_binary.shape
    device = seq_binary.device
    t = torch.linspace(0, 1, L, device=device)
    envelope = torch.exp(-fading * t)
    amplitude = seq_binary.float() * envelope.unsqueeze(0)
    return amplitude
def calculate_physics_metrics(amplitude_seq, target_len, n_fft):
    B = amplitude_seq.shape[0]
    if isinstance(target_len, int):
        lens = torch.full((B,), target_len, device=amplitude_seq.device)
    else:
        lens = target_len.to(amplitude_seq.device)
    max_l = int(lens.max().item())
    seq = amplitude_seq[:, :max_l].clone()
    for i in range(B):
        l = int(lens[i].item())
        if l < max_l:
            seq[i, l:] = 0.0
    X = torch.fft.fft(seq, n=n_fft, dim=-1)
    spec_inv = torch.abs(torch.reciprocal(X + 1e-12)) ** 2
    Q = torch.mean(spec_inv, dim=-1)
    gc = 10 * torch.log10(torch.sqrt(1.0 / (Q + 1e-12)))
    fe = seq.sum(dim=-1)
    gr = 10 * torch.log10(torch.sqrt(fe / 2.0 + 1e-12))
    return gc, gr, fe, lens
# ====================== 模型定义 ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
    def forward(self, x):
        return F.gelu(x + self.net(x))
class UniversalCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_embed = nn.Linear(1, HIDDEN_DIM // 2)
        self.pos_enc = PositionalEncoding(HIDDEN_DIM // 2)
        self.cond_embed = nn.Linear(COND_DIM, HIDDEN_DIM // 2)
        self.encoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.GELU(),
            ResBlock(HIDDEN_DIM), ResBlock(HIDDEN_DIM), ResBlock(HIDDEN_DIM),
        )
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.dec_input = nn.Linear(LATENT_DIM + COND_DIM, HIDDEN_DIM)
        self.decoder = nn.Sequential(
            ResBlock(HIDDEN_DIM), ResBlock(HIDDEN_DIM), ResBlock(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, MAX_LEN),
        )
    def encode(self, x, c):
        B, L = x.shape
        x_emb = self.seq_embed(x.unsqueeze(-1))
        x_emb = self.pos_enc(x_emb)
        c_emb = self.cond_embed(c).unsqueeze(1).expand(-1, L, -1)
        h = torch.cat([x_emb, c_emb], dim=-1)
        h = h.mean(dim=1)
        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    def decode(self, z, c, temperature=1.0):
        h = torch.cat([z, c], dim=1)
        h = self.dec_input(h)
        h = self.decoder(h)
        return torch.sigmoid(h / temperature)
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar
class CheckpointManager:
    def __init__(self, ckpt_dir, name="universal_cvae_v2"):
        self.best_path = os.path.join(ckpt_dir, f"{name}_best.pt")
        self.last_path = os.path.join(ckpt_dir, f"{name}_last.pt")
    def load(self, model, prefer_best=True):
        path = self.best_path if prefer_best else self.last_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到检查点：{path}")
        print(f"加载模型：{path}")
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        return model
# ====================== 生成函数（改进1：取消随机化）======================
def generate_for_fading(model, fe, fading, n_sample, n_rounds, temperature,
                        top_k, output_base, target_ratio=1.0,
                        amp_sum_tolerance=None):
    target_len = M_FACTOR * fe
    gr0 = 10 * np.log10(np.sqrt(fe / 2.0))
    print(f"\n生成参数：理论 FE={fe}, 码长={target_len}, Gr0={gr0:.3f} dB, 衰落={fading:.2f}")
    print(f"采样：{n_rounds} 轮 × {n_sample} 条，温度={temperature}")
    print(f"二值化策略：模型自行决定1的个数（概率和四舍五入）")
    if amp_sum_tolerance:
        print(f"后筛选：幅度和误差 ≤ {amp_sum_tolerance}")
    else:
        print("后筛选：仅按 Diff 排序，不限制幅度和")
    cond = make_condition(fe, target_len, fading).to(DEVICE)
    all_codes = []
    batch = 200
    with torch.no_grad():
        for rnd in range(n_rounds):
            rnd_codes = []
            rnd_prob_sums = []
            for start in range(0, n_sample, batch):
                bs = min(batch, n_sample - start)
                z = torch.randn(bs, LATENT_DIM).to(DEVICE)
                c_b = cond.unsqueeze(0).expand(bs, -1)
                probs = model.decode(z, c_b, temperature=temperature)
                probs_eff = probs[:, :target_len].cpu()
                # ---- 改进1：用概率和四舍五入决定1的个数 ----
                codes = torch.zeros(bs, target_len)
                actual_fe_list = []
                for i in range(bs):
                    p = probs_eff[i]
                    n_ones = int(torch.round(p.sum()).clip(1, target_len).item())
                    actual_fe_list.append(n_ones)
                    _, idx = torch.sort(p, descending=True)
                    top_idx = idx[:n_ones]
                    codes[i, top_idx] = 1.0
                rnd_codes.append(codes)
                rnd_prob_sums.extend(actual_fe_list)
            rnd_codes = torch.cat(rnd_codes)
            bin_fe = rnd_codes.sum(dim=1).numpy()
            sample_amp = apply_edfa_envelope(rnd_codes.to(DEVICE), fading=fading)
            amp_sum = sample_amp.sum(dim=1).cpu().numpy()
            print(f"  轮次 {rnd+1}: 二进制FE={bin_fe.mean():.1f}±{bin_fe.std():.1f} | "
                  f"衰落后幅度和={amp_sum.mean():.1f}±{amp_sum.std():.1f}")
            all_codes.append(rnd_codes)
    pool_codes = torch.cat(all_codes)
    print(f"\n候选池：{len(pool_codes)} 条，高精度 FFT（{target_len * 200} 点）...")
    # 高精度评估
    final_gc_l, final_gr_l, final_fe_l = [], [], []
    eval_batch = 100
    n_fft_final = target_len * 200
    for start in range(0, len(pool_codes), eval_batch):
        chunk = pool_codes[start:start+eval_batch].to(DEVICE)
        amplitude = apply_edfa_envelope(chunk, fading=fading)
        chunk_padded = F.pad(amplitude, (0, MAX_LEN - target_len))
        gc_b, gr_b, fe_b, _ = calculate_physics_metrics(chunk_padded, target_len, n_fft_final)
        final_gc_l.append(gc_b.cpu())
        final_gr_l.append(gr_b.cpu())
        final_fe_l.append(fe_b.cpu())
    final_gc = torch.cat(final_gc_l)
    final_gr = torch.cat(final_gr_l)
    final_fe = torch.cat(final_fe_l)
    final_diff = final_gc - final_gr
    # 按 Diff 排序去重
    sorted_idx = torch.argsort(final_diff, descending=True)
    seen, unique_idx = set(), []
    for idx in sorted_idx:
        key = tuple(pool_codes[idx].numpy().astype(int).tolist())
        if key not in seen:
            seen.add(key)
            unique_idx.append(idx)
        if len(unique_idx) >= top_k * 3:
            break
    # 后筛选：根据幅度和过滤
    target_amp = fe * target_ratio
    if amp_sum_tolerance is not None:
        filtered_idx = []
        for idx in unique_idx:
            amp = final_fe[idx].item()
            if abs(amp - target_amp) <= amp_sum_tolerance:
                filtered_idx.append(idx)
            if len(filtered_idx) >= top_k:
                break
        if len(filtered_idx) > 0:
            unique_idx = filtered_idx
        else:
            print(f"警告：没有序列的幅度和满足误差 {amp_sum_tolerance}，使用原始排序结果。")
    # 创建输出文件夹
    fading_tag = f"fading{fading:.2f}".replace('.', '_')
    out_dir = os.path.join(output_base, f"FE{fe}_L{target_len}", fading_tag)
    os.makedirs(out_dir, exist_ok=True)
    # 清空旧文件
    for old_f in os.listdir(out_dir):
        if old_f.endswith(".txt"):
            os.remove(os.path.join(out_dir, old_f))
    print(f"\n{'排名':<5} {'二进制FE':<8} {'幅度和':<8} {'Gc':>8} {'Gr':>8} {'Gr0':>8} {'Diff':>8} {'|Gr-Gr0|':>10}")
    print("-" * 80)
    rows = []
    exported = 0
    for rank, idx in enumerate(unique_idx[:top_k]):
        idx = idx.item()
        seq = pool_codes[idx].numpy().astype(int)
        gc_v = final_gc[idx].item()
        gr_v = final_gr[idx].item()
        amp_sum_v = final_fe[idx].item()
        bin_fe_v = int(seq.sum())
        dv = final_diff[idx].item()
        gr_err = abs(gr_v - gr0)
        fname = (f"Rank{rank:02d}_BinFE{bin_fe_v}_AmpSum{amp_sum_v:.1f}"
                 f"_Gc{gc_v:.3f}_Gr{gr_v:.3f}_Diff{dv:+.3f}_Gr0{gr0:.3f}.txt")
        with open(os.path.join(out_dir, fname), 'w') as f:
            f.write("".join(map(str, seq)))
        flag = "  ← Gr偏>1dB" if gr_err > 1.0 else ""
        print(f"  {rank:02d}   {bin_fe_v:<8} {amp_sum_v:<8.1f} {gc_v:>8.3f} {gr_v:>8.3f} "
              f"{gr0:>8.3f} {dv:>+8.3f} {gr_err:>10.3f}{flag}")
        exported += 1
        rows.append({
            "FE_theory": fe,
            "Fading": fading,
            "Rank": rank+1,
            "Binary_FE": bin_fe_v,
            "AmpSum": amp_sum_v,
            "Gc": gc_v,
            "Gr": gr_v,
            "Diff": dv,
            "Gr0": gr0,
            "|Gr-Gr0|": gr_err
        })
    best_i = unique_idx[0].item()
    print(f"\n✅ 导出 {exported} 条 → {out_dir}")
    print(f"   最优：Gc={final_gc[best_i]:.3f}  Gr={final_gr[best_i]:.3f}  Diff={final_diff[best_i]:+.3f}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return pd.DataFrame(rows) if rows else None

# ====================== 批量生成主函数 ======================
def batch_generate():
    parser = argparse.ArgumentParser(description="批量生成：改进版（模型自定FE个数）")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output_base", type=str, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--fe_list", type=int, nargs='+', default=None)
    parser.add_argument("--fading_list", type=float, nargs='+', default=None)
    parser.add_argument("--n_sample", type=int, default=20000)
    parser.add_argument("--n_rounds", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--target_ratio", type=float, default=1.0,
                        help="目标幅度和与理论 FE 的比例（默认1.0）")
    parser.add_argument("--amp_sum_tolerance", type=float, default=None,
                        help="幅度和后筛选允许的绝对误差（例如 2.0），若为None则不筛选")
    args = parser.parse_args()

    if args.fe_list is None:
        fe_list = list(range(40, 221, 10))
    else:
        fe_list = args.fe_list
    if args.fading_list is None:
        fading_list = [0.0, 0.15]#[0.0, 0.15, 0.25, 0.35, 0.5]
    else:
        fading_list = args.fading_list

    print("批量生成参数：")
    print(f"  FE 列表: {fe_list}")
    print(f"  衰落列表: {fading_list}")
    print(f"  采样: {args.n_sample}×{args.n_rounds} 条/组合, top_k={args.top_k}, temp={args.temperature}")
    print(f"  二值化策略: 模型自定个数 (概率和四舍五入)")
    if args.amp_sum_tolerance:
        print(f"  后筛选: 幅度和误差 ≤ {args.amp_sum_tolerance}")

    # 加载模型
    print("\n加载模型...")
    model = UniversalCVAE().to(DEVICE)
    ckpt_mgr = CheckpointManager(args.checkpoint_dir)
    model = ckpt_mgr.load(model, prefer_best=True)
    model.eval()
    print("模型加载完成\n")

    total = len(fe_list) * len(fading_list)
    count = 0
    all_summaries = []
    for fe in fe_list:
        for fading in fading_list:
            count += 1
            print(f"\n========== 进度 {count}/{total} ==========")
            df = generate_for_fading(
                model=model, fe=fe, fading=fading,
                n_sample=args.n_sample, n_rounds=args.n_rounds,
                temperature=args.temperature, top_k=args.top_k,
                output_base=args.output_base,
                target_ratio=args.target_ratio,
                amp_sum_tolerance=args.amp_sum_tolerance
            )
            if df is not None:
                all_summaries.append(df)

    if all_summaries:
        master_df = pd.concat(all_summaries, ignore_index=True)
        master_df.to_csv(os.path.join(args.output_base, "all_results.csv"), index=False)
        print(f"\n所有结果汇总已保存: {os.path.join(args.output_base, 'all_results.csv')}")

    print("\n批量生成完成！")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    batch_generate()
