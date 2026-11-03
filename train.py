import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import json
import argparse
import torch.fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from collections import defaultdict
# ====================== 配置 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M_FACTOR = 3
MAX_LEN = 1024
LATENT_DIM = 128
HIDDEN_DIM = 1024
COND_DIM = 3
MAX_FE = 700
MAX_FADING = 0.5
BATCH_SIZE = 32
EPOCHS = 900
LR = 1e-4
# 损失权重（调整后）
W_KLD = 0.2
W_GC = 0.5
W_GR0 = 0.5
W_DIFF = 1.0
W_FE = 0.5
GR_DELTA = 0.5
BCE_START = 1.0
BCE_END = 0.5          # 不要降太低
BCE_DECAY_EP = 400
PHYS_START_EP = 30     # 提前引入物理约束
PHYS_MAX_W = 0.5       # 物理损失最大权重（不要过大）
DIFF_SCALE_START_EP = 150
DIFF_SCALE_END_EP = 300
DIFF_SCALE_MAX = 2.0
# 数据与输出路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data0")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "outputs")
# ====================== 工具函数 ======================
def apply_edfa_envelope(seq_binary: torch.Tensor, fading: float) -> torch.Tensor:
    """对二进制序列应用包络，返回衰落后的幅度序列"""
    if fading <= 0:
        return seq_binary.float()
    B, L = seq_binary.shape
    device = seq_binary.device
    t = torch.linspace(0, 1, L, device=device)
    envelope = torch.exp(-fading * t)
    amplitude = seq_binary.float() * envelope.unsqueeze(0)
    return amplitude
def make_condition(fe: int, target_len: int, fading: float) -> torch.Tensor:
    fe_norm = fe / MAX_FE
    len_norm = target_len / MAX_LEN
    fading_norm = fading / MAX_FADING
    return torch.FloatTensor([fe_norm, len_norm, fading_norm])
def decode_len_from_cond(c: torch.Tensor) -> torch.Tensor:
    return (c[:, 1] * MAX_LEN).round().long()
def gr0_from_fe(fe):
    if isinstance(fe, torch.Tensor):
        return 10 * torch.log10(torch.sqrt(fe / 2.0 + 1e-12))
    else:
        return 10 * np.log10(np.sqrt(fe / 2.0 + 1e-12))
def bce_weight(epoch: int) -> float:
    if epoch >= BCE_DECAY_EP:
        return BCE_END
    t = epoch / BCE_DECAY_EP
    return BCE_START * (1 - t) + BCE_END * t
def phys_weight(epoch: int) -> float:
    if epoch < PHYS_START_EP:
        return 0.0
    return min(PHYS_MAX_W, (epoch - PHYS_START_EP) / (EPOCHS - PHYS_START_EP))
def calculate_physics_metrics(amplitude_seq, target_len, n_fft):
    """输入已经是衰落后的幅度序列，直接计算物理指标"""
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
    fe = seq.sum(dim=-1)          # 衰落后的幅度和（等效FE）
    gr = 10 * torch.log10(torch.sqrt(fe / 2.0 + 1e-12))
    return gc, gr, fe, lens
# ====================== 模型定义（与之前相同）======================
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
# ====================== 数据集（关键修改：输入为衰落后的序列）======================
class ConditionalFadingDataset(Dataset):
    def __init__(self, data_root, max_len=MAX_LEN):
        self.samples = []
        if not os.path.exists(data_root):
            print(f"[错误] 数据根目录不存在：{data_root}")
            return
        fading_dirs = [d for d in os.listdir(data_root) if d.startswith('fading_')]
        if not fading_dirs:
            print(f"[错误] 未找到 fading_* 子文件夹")
            return
        total_loaded = 0
        total_skipped = 0
        for folder_name in fading_dirs:
            try:
                fading = float(folder_name.split('_')[1])
            except:
                print(f"[警告] 无法解析文件夹名：{folder_name}，跳过")
                continue
            folder_path = os.path.join(data_root, folder_name)
            txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            print(f"处理 {folder_name} (fading={fading:.2f})，共 {len(txt_files)} 个文件")
            for fn in tqdm(txt_files, desc=folder_name, leave=False):
                file_path = os.path.join(folder_path, fn)
                try:
                    with open(file_path, 'r') as f:
                        raw = f.read().strip()
                    bits = [int(c) for c in raw if c in '01']
                    seq_len = len(bits)
                    if seq_len < 10 or seq_len > max_len:
                        total_skipped += 1
                        continue
                    actual_fe = sum(bits)
                    # 原始二进制目标 y
                    y = torch.zeros(max_len, dtype=torch.float32)
                    y[:seq_len] = torch.tensor(bits, dtype=torch.float32)
                    # 输入 x_faded：对原始二进制应用包络
                    y_seq = y[:seq_len].unsqueeze(0)  # (1, seq_len)
                    x_faded_seq = apply_edfa_envelope(y_seq, fading)  # (1, seq_len)
                    x_faded = torch.zeros(max_len, dtype=torch.float32)
                    x_faded[:seq_len] = x_faded_seq.squeeze(0)
                    # 条件向量
                    cond = make_condition(actual_fe, seq_len, fading)
                    self.samples.append({
                        'x': x_faded,
                        'y': y,
                        'cond': cond,
                        'seq_len': seq_len,
                        'fe': actual_fe,
                        'fading': fading
                    })
                    total_loaded += 1
                except Exception as e:
                    print(f"[警告] 读取 {file_path} 失败：{e}")
                    total_skipped += 1
        print(f"\n数据集统计：")
        print(f"  加载成功: {total_loaded} 条")
        print(f"  跳过: {total_skipped} 条")
        if total_loaded > 0:
            fading_counts = defaultdict(int)
            for s in self.samples:
                fading_counts[s['fading']] += 1
            print("  各衰落强度样本数：")
            for f, cnt in sorted(fading_counts.items()):
                print(f"    fading={f:.2f}: {cnt} 条")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s['x'],          # 衰落后的幅度序列（输入）
            s['y'],          # 原始二进制（目标）
            s['cond'],
            torch.LongTensor([s['seq_len']]),
            torch.FloatTensor([s['fe']]),
            torch.FloatTensor([s['fading']])
        )
# ====================== 损失函数（关键修改：对输出应用包络再计算物理指标）======================
def compute_loss(recon, y, mu, logvar, c, lens, target_fe_batch, fading_batch, epoch):
    B = recon.shape[0]
    w_bce = bce_weight(epoch)
    w_phys = phys_weight(epoch)
    # BCE 损失（在原始二进制上，未衰落）
    bce = 0.0
    for i in range(B):
        l = int(lens[i].item())
        bce += F.binary_cross_entropy(recon[i, :l], y[i, :l])
    bce = bce / B
    # KL 散度
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # 计算物理指标前，先对重构概率应用包络（模拟衰落后的信号）
    # 注意：recon 是概率（0~1），应用包络后作为幅度
    recon_faded = recon.clone()
    t_lens = decode_len_from_cond(c)
    n_fft = max(4096, int(t_lens.max().item()) * 50)
    for i in range(B):
        l = int(t_lens[i].item())
        fading_i = fading_batch[i].item()
        if fading_i > 0:
            t = torch.linspace(0, 1, l, device=recon.device)
            envelope = torch.exp(-fading_i * t)
            recon_faded[i, :l] = recon[i, :l] * envelope
    gc, gr, fe_out, _ = calculate_physics_metrics(recon_faded, t_lens, n_fft)
    target_fe = target_fe_batch.squeeze(1).float().to(recon.device)
    gr0_target = gr0_from_fe(target_fe)
    loss_gc = -gc.mean()
    loss_gr0 = F.mse_loss(gr, gr0_target)
    loss_diff = F.huber_loss(gc - gr, torch.zeros_like(gc), delta=GR_DELTA)
    fe_loss = F.mse_loss(fe_out, target_fe) / (MAX_FE ** 2)
    # Diff 缩放
    if epoch < DIFF_SCALE_START_EP:
        diff_scale = 1.0
    elif epoch >= DIFF_SCALE_END_EP:
        diff_scale = DIFF_SCALE_MAX
    else:
        t = (epoch - DIFF_SCALE_START_EP) / (DIFF_SCALE_END_EP - DIFF_SCALE_START_EP)
        diff_scale = 1.0 + t * (DIFF_SCALE_MAX - 1.0)
    loss_diff_scaled = loss_diff * diff_scale
    phys_loss = (W_GC * loss_gc + W_GR0 * loss_gr0 + W_DIFF * loss_diff_scaled + W_FE * fe_loss)
    loss = w_bce * bce + W_KLD * kld + w_phys * phys_loss
    return loss, bce.item(), kld.item(), gc.mean().item(), gr.mean().item(), fe_out.mean().item(), loss_gc.item(), loss_gr0.item(), loss_diff.item(), diff_scale
# ====================== 检查点管理器 ======================
class CheckpointManager:
    def __init__(self, ckpt_dir, name="universal_cvae_v2"):
        self.dir = ckpt_dir
        self.name = name
        self.best_path = os.path.join(ckpt_dir, f"{name}_best.pt")
        self.last_path = os.path.join(ckpt_dir, f"{name}_last.pt")
        self.meta_path = os.path.join(ckpt_dir, f"{name}_meta.json")
        self.history_dir = os.path.join(ckpt_dir, "history")
        os.makedirs(self.history_dir, exist_ok=True)
        self.best_diff = -float("inf")
        os.makedirs(ckpt_dir, exist_ok=True)
    def save(self, model, optimizer, epoch, val_diff, gc_hist, gr_hist, is_best):
        state = {
            "epoch": epoch, "val_diff": val_diff,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "gc_hist": gc_hist, "gr_hist": gr_hist,
            "hparams": {"MAX_LEN": MAX_LEN, "LATENT_DIM": LATENT_DIM,
                        "HIDDEN_DIM": HIDDEN_DIM, "COND_DIM": COND_DIM, "MAX_FE": MAX_FE, "MAX_FADING": MAX_FADING},
        }
        torch.save(state, self.last_path)
        if is_best:
            torch.save(state, self.best_path)
            self.best_diff = val_diff
            print(f"  ✅ 保存最优模型（val_diff={val_diff:.4f} dB，Epoch {epoch}）")
            history_path = os.path.join(self.history_dir, f"best_epoch{epoch:04d}_diff{val_diff:+.4f}.pt")
            torch.save(state, history_path)
            print(f"  📁 已保存历史最佳模型: {history_path}")
        with open(self.meta_path, "w") as f:
            json.dump({"epoch": epoch, "val_diff": val_diff, "best_diff": self.best_diff}, f, indent=2)
    def load(self, model, optimizer=None, prefer_best=False):
        path = self.best_path if prefer_best else self.last_path
        if not os.path.exists(path):
            print("[CheckpointManager] 未找到检查点，从头训练。")
            return 0, [], []
        print(f"[CheckpointManager] 加载：{path}")
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        self.best_diff = ckpt.get("val_diff", -float("inf"))
        return ckpt.get("epoch", 0), ckpt.get("gc_hist", []), ckpt.get("gr_hist", [])
    def exists(self):
        return os.path.exists(self.last_path)
# ====================== 验证函数（同样对二值化序列应用包络）======================
@torch.no_grad()
def validate(model, val_loader, epoch):
    model.eval()
    len_stats = defaultdict(lambda: {"gc": [], "gr": [], "diff": [], "fe_err": []})
    for x, y, c, lens, target_fe_b, fading_b in val_loader:
        x, y, c = x.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
        lens = lens.squeeze(1).to(DEVICE)
        target_fe_b = target_fe_b.to(DEVICE)
        fading_b = fading_b.to(DEVICE)
        recon, _, _ = model(x, c)
        t_lens = decode_len_from_cond(c)
        # 二值化：取概率最高的 tfe 个位置为1
        probs_cpu = recon.cpu()
        bin_list = []
        for i in range(recon.shape[0]):
            tfe = int(target_fe_b[i].item())
            tlen = int(t_lens[i].item())
            p = probs_cpu[i, :tlen]
            _, idx = torch.sort(p, descending=True)
            top_idx = idx[:tfe]
            b = torch.zeros(tlen)
            b[top_idx] = 1.0
            padded = torch.zeros(MAX_LEN)
            padded[:tlen] = b
            bin_list.append(padded)
        bin_r = torch.stack(bin_list).to(DEVICE)
        # 对二值化序列应用包络，再计算物理指标
        bin_faded = bin_r.clone()
        for i in range(bin_r.shape[0]):
            l = int(t_lens[i].item())
            fading_i = fading_b[i].item()
            if fading_i > 0:
                t = torch.linspace(0, 1, l, device=DEVICE)
                envelope = torch.exp(-fading_i * t)
                bin_faded[i, :l] = bin_r[i, :l] * envelope
        n_fft_val = max(4096, int(t_lens.max().item()) * 20)
        gc, gr, fe_out, _ = calculate_physics_metrics(bin_faded, t_lens, n_fft_val)
        for i in range(len(t_lens)):
            seq_len = t_lens[i].item()
            len_stats[seq_len]["gc"].append(gc[i].item())
            len_stats[seq_len]["gr"].append(gr[i].item())
            len_stats[seq_len]["diff"].append((gc[i] - gr[i]).item())
            len_stats[seq_len]["fe_err"].append((fe_out[i] - target_fe_b[i]).abs().item())
    torch.cuda.empty_cache()
    all_gc = [g for stats in len_stats.values() for g in stats["gc"]]
    all_gr = [g for stats in len_stats.values() for g in stats["gr"]]
    all_diff = [d for stats in len_stats.values() for d in stats["diff"]]
    all_fe_err = [e for stats in len_stats.values() for e in stats["fe_err"]]
    gc_m = np.mean(all_gc) if all_gc else 0.0
    gr_m = np.mean(all_gr) if all_gr else 0.0
    diff_m = np.mean(all_diff) if all_diff else 0.0
    fe_err_m = np.mean(all_fe_err) if all_fe_err else 0.0
    return gc_m, gr_m, diff_m, fe_err_m, len_stats
# ====================== 训练函数 ======================
def train(resume=False):
    dataset = ConditionalFadingDataset(DATA_ROOT)
    if len(dataset) == 0:
        print("[错误] 数据集为空")
        return
    val_n = max(1, int(len(dataset) * 0.1))
    train_n = len(dataset) - val_n
    train_set, val_set = random_split(dataset, [train_n, val_n], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"训练集 {train_n} 条，验证集 {val_n} 条\n")
    model = UniversalCVAE().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    ckpt_mgr = CheckpointManager(CHECKPOINT_DIR)
    start_epoch, gc_hist, gr_hist = 0, [], []
    if resume and ckpt_mgr.exists():
        start_epoch, gc_hist, gr_hist = ckpt_mgr.load(model, optimizer)
    best_diff = ckpt_mgr.best_diff
    plot_dir = os.path.join(CHECKPOINT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"设备：{DEVICE}")
    print(f"条件维度：{COND_DIM} (FE, 码长, fading)")
    print(f"损失权重：W_GC={W_GC}, W_GR0={W_GR0}, W_DIFF={W_DIFF}, W_FE={W_FE}")
    print(f"BCE衰减：{BCE_START}→{BCE_END}（{BCE_DECAY_EP} epoch）")
    print(f"物理损失权重：epoch {PHYS_START_EP} 开始线性增至 {PHYS_MAX_W}")
    print(f"Diff缩放：epoch {DIFF_SCALE_START_EP}→{DIFF_SCALE_END_EP} 线性增至 {DIFF_SCALE_MAX} 倍")
    print(f"训练：输入为衰落后的幅度序列，目标为原始二进制，物理损失基于衰落后的重构信号")
    print()
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        train_loss_avg = 0.0
        for x, y, c, lens, target_fe_b, fading_b in train_loader:
            x, y, c = x.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
            lens = lens.squeeze(1).to(DEVICE)
            target_fe_b = target_fe_b.to(DEVICE)
            fading_b = fading_b.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(x, c)
            loss, bce, kld, gc, gr, fe_out, loss_gc, loss_gr0, loss_diff, diff_scale = compute_loss(
                recon, y, mu, logvar, c, lens, target_fe_b, fading_b, epoch
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_avg += loss.item()
        scheduler.step()
        train_loss_avg /= len(train_loader)
        # 验证
        gc_m, gr_m, diff_m, fe_err_m, len_stats = validate(model, val_loader, epoch)
        gc_hist.append(gc_m)
        gr_hist.append(gr_m)
        is_best = diff_m > best_diff + 0.001
        if is_best:
            best_diff = diff_m
        ckpt_mgr.save(model, optimizer, epoch, diff_m, gc_hist, gr_hist, is_best)
        if epoch % 10 == 0 or is_best:
            lr_now = optimizer.param_groups[0]['lr']
            w_phys = phys_weight(epoch)
            marker = " ← BEST" if is_best else ""
            print(f"Epoch {epoch:4d} | Loss={train_loss_avg:.4f} | "
                  f"Gc={gc_m:.3f} Gr={gr_m:.3f} Diff={diff_m:+.3f} | "
                  f"FE_err={fe_err_m:.1f} | w_bce={bce_weight(epoch):.3f} "
                  f"w_phys={w_phys:.3f} diff_scale={diff_scale:.2f} LR={lr_now:.2e}{marker}")
            print(f"\n  --- 按码长验证结果 (Epoch {epoch}) ---")
            for seq_len in sorted(len_stats.keys()):
                stats = len_stats[seq_len]
                gc_len = np.mean(stats["gc"])
                gr_len = np.mean(stats["gr"])
                diff_len = np.mean(stats["diff"])
                fe_err_len = np.mean(stats["fe_err"])
                print(f"  码长 {seq_len:4d} (FE≈{seq_len//M_FACTOR:3d}): "
                      f"Gc={gc_len:.3f} Gr={gr_len:.3f} Diff={diff_len:+.3f} FE_err={fe_err_len:.1f}")
            print()
        if epoch % 50 == 0:
            _save_plot(gc_hist, gr_hist, epoch, plot_dir)
    print("\n训练完成。")
def _save_plot(gc_hist, gr_hist, epoch, plot_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = range(1, len(gc_hist) + 1)
    ax.plot(xs, gc_hist, label='Gc', color='#185FA5', linewidth=1)
    ax.plot(xs, gr_hist, label='Gr', color='#0F6E56', linewidth=1)
    diff_hist = [g - r for g, r in zip(gc_hist, gr_hist)]
    ax.plot(xs, diff_hist, label='Diff=Gc-Gr', color='#D85A30', linewidth=1, linestyle='--')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_title(f"Epoch {epoch} — Diff > 0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("dB")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, f"epoch_{epoch:04d}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [图表] 已保存 → {path}")
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="条件 CVAE 训练（修正版：输入衰落序列，物理损失应用包络）")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data_root", "--data-root", default=DATA_ROOT)
    parser.add_argument("--checkpoint_dir", "--checkpoint-dir", default=CHECKPOINT_DIR)
    args = parser.parse_args()
    DATA_ROOT = args.data_root
    CHECKPOINT_DIR = args.checkpoint_dir
    train(resume=args.resume)
