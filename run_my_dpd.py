"""
run_my_dpd.py  —  Neural DPD training & evaluation on real dpdOpen data.

Place at project root. Run:
    python run_my_dpd.py --dataset DPA_200MHz --backbone my_lstm_dpd --epochs 100
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

import opendpd
from backbones.my_dpd_backbone import MyMLPDPD, MyLSTMDPD

BACKBONE_MAP = {
    'my_mlp_dpd':  MyMLPDPD,
    'my_lstm_dpd': MyLSTMDPD,
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',  default='DPA_200MHz')
    p.add_argument('--backbone', default='my_lstm_dpd', choices=list(BACKBONE_MAP.keys()))
    p.add_argument('--hidden',   type=int,   default=32)
    p.add_argument('--seq_len',  type=int,   default=15)
    p.add_argument('--epochs',   type=int,   default=100)
    p.add_argument('--batch',    type=int,   default=512)
    p.add_argument('--lr',       type=float, default=5e-4)
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir',  default='results_my_dpd')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def to_complex(arr):
    return arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64)


def make_sequences(X, y, seq_len):
    """Sliding window: X (N,2) -> (M, seq_len, 2),  y (N,2) -> (M, 2)"""
    Xw = np.lib.stride_tricks.sliding_window_view(X, (seq_len, 2))[:, 0]
    yw = y[seq_len - 1:]
    return Xw.astype(np.float32), yw.astype(np.float32)


def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def evm_percent(ref, test):
    return float(
        np.sqrt(np.mean(np.abs(test - ref)**2)) /
        np.sqrt(np.mean(np.abs(ref)**2)) * 100
    )

def nmse_db(ref, test):
    return float(10 * np.log10(
        np.mean(np.abs(test - ref)**2) / (np.mean(np.abs(ref)**2) + 1e-15)
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += criterion(model(xb), yb).item() * len(xb)
    return total / len(loader.dataset)

@torch.no_grad()
def predict_complex(model, X_seq, batch_size, device):
    model.eval()
    preds = []
    for i in range(0, len(X_seq), batch_size):
        xb = torch.from_numpy(X_seq[i:i+batch_size]).to(device)
        preds.append(model(xb).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    return pred[:, 0] + 1j * pred[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Academic-style plotting
# ─────────────────────────────────────────────────────────────────────────────
def get_psd(sig_c, nperseg=512):
    from scipy.signal import welch
    f, P = welch(sig_c.real, nperseg=nperseg, return_onesided=False)
    return np.fft.fftshift(f), 10 * np.log10(np.fft.fftshift(P) + 1e-15)


def academic_axes(ax, xlabel='', ylabel='', title=''):
    ax.set_facecolor('white')
    ax.spines[:].set_color('black')
    ax.spines[:].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', direction='in',
                   length=4, width=0.8, labelsize=8, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in',
                   length=2, width=0.6, top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(xlabel, fontsize=9, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=4)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', pad=5)
    ax.grid(True, which='major', linestyle='--', linewidth=0.4,
            color='#cccccc', zorder=0)


def make_plots(args, train_losses, val_losses,
               ref_c, pa_c, dpd_c,
               evm_before, evm_after,
               nmse_before, nmse_after,
               out_dir):

    C_REF = '#444444'
    C_PA  = '#d62728'
    C_DPD = '#1f77b4'
    C_TR  = '#2ca02c'
    C_VA  = '#ff7f0e'

    plt.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.linewidth':   0.8,
        'figure.dpi':       150,
    })

    backbone_label = args.backbone.replace('_', '-').upper()

    fig = plt.figure(figsize=(14, 10), facecolor='white')
    fig.suptitle(
        f'Neural Digital Pre-Distortion: {backbone_label} on {args.dataset}\n'
        f'Hidden size: {args.hidden}  |  Memory depth: {args.seq_len}  |  Epochs: {args.epochs}',
        fontsize=10, fontweight='bold', y=0.99, va='top'
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.52, wspace=0.40,
                           left=0.08, right=0.97,
                           top=0.93, bottom=0.07)

    N = min(len(ref_c), len(pa_c), len(dpd_c), 20_000)

    # ── (a) EVM & NMSE grouped bar ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    academic_axes(ax, ylabel='Value', title='(a) EVM and NMSE Comparison')

    x     = np.array([0.0, 1.0])
    width = 0.35
    b1 = ax.bar(x - width/2, [evm_before, abs(nmse_before)],
                width, label='Without DPD', color=C_PA,
                edgecolor='black', linewidth=0.6, zorder=3)
    b2 = ax.bar(x + width/2, [evm_after,  abs(nmse_after)],
                width, label='With DPD',    color=C_DPD,
                edgecolor='black', linewidth=0.6, zorder=3)

    for bar, val, unit in zip(list(b1) + list(b2),
                               [evm_before, abs(nmse_before), evm_after, abs(nmse_after)],
                               ['%', 'dB', '%', 'dB']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{val:.1f}{unit}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['EVM (%)', '|NMSE| (dB)'], fontsize=8)
    ax.legend(fontsize=7.5, frameon=True, edgecolor='black', fancybox=False)
    ax.set_axisbelow(True)

    # ── (b) PSD ───────────────────────────────────────────────────────────────
    ax_psd = fig.add_subplot(gs[0, 1:])
    academic_axes(ax_psd,
                  xlabel=r'Normalised Frequency ($f\,/\,f_s$)',
                  ylabel='PSD (dBW/Hz)',
                  title='(b) Power Spectral Density — Spectral Regrowth')

    f_r, p_r = get_psd(ref_c[:N])
    f_p, p_p = get_psd(pa_c[:N])
    f_d, p_d = get_psd(dpd_c[:N])

    ax_psd.plot(f_r, p_r, color=C_REF, lw=1.0, ls='-',  label='Input (reference)', zorder=3)
    ax_psd.plot(f_p, p_p, color=C_PA,  lw=1.2, ls='-',  label='PA output (no DPD)', zorder=4)
    ax_psd.plot(f_d, p_d, color=C_DPD, lw=1.2, ls='--', label=f'{backbone_label} + PA', zorder=5)
    ax_psd.legend(fontsize=7.5, frameon=True, edgecolor='black', fancybox=False)
    ax_psd.set_xlim(-0.5, 0.5)

    # ── (c) Training curve ────────────────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[1, 0])
    academic_axes(ax_loss, xlabel='Epoch', ylabel='MSE Loss',
                  title='(c) Training and Validation Loss')
    epochs = np.arange(1, len(train_losses) + 1)
    ax_loss.semilogy(epochs, train_losses, color=C_TR, lw=1.2, label='Training')
    ax_loss.semilogy(epochs, val_losses,   color=C_VA, lw=1.2, ls='--', label='Validation')
    ax_loss.legend(fontsize=7.5, frameon=True, edgecolor='black', fancybox=False)

    # ── (d) AM/AM ─────────────────────────────────────────────────────────────
    ax_amam = fig.add_subplot(gs[1, 1])
    academic_axes(ax_amam,
                  xlabel='Normalised Input Amplitude',
                  ylabel='Normalised Output Amplitude',
                  title='(d) AM/AM Characteristic')

    idx = np.random.choice(N, min(3000, N), replace=False)
    a_in  = np.abs(ref_c[idx]);  a_in_n  = a_in  / (a_in.max()  + 1e-10)
    a_pa  = np.abs(pa_c[idx]);   a_pa_n  = a_pa  / (a_pa.max()  + 1e-10)
    a_dpd = np.abs(dpd_c[idx]);  a_dpd_n = a_dpd / (a_dpd.max() + 1e-10)
    s = np.argsort(a_in_n)

    ax_amam.plot(a_in_n[s], a_in_n[s], color=C_REF, lw=1.0, ls='--',
                 label='Ideal (linear)', zorder=5)
    ax_amam.scatter(a_in_n[s][::2], a_pa_n[s][::2],  s=1.5, alpha=0.4,
                    color=C_PA,  label='PA (no DPD)', zorder=3)
    ax_amam.scatter(a_in_n[s][::2], a_dpd_n[s][::2], s=1.5, alpha=0.4,
                    color=C_DPD, label=f'{backbone_label}', zorder=4)
    ax_amam.legend(fontsize=7.5, frameon=True, edgecolor='black',
                   fancybox=False, markerscale=4)

    # ── (e) AM/PM ─────────────────────────────────────────────────────────────
    ax_ampm = fig.add_subplot(gs[1, 2])
    academic_axes(ax_ampm,
                  xlabel='Normalised Input Amplitude',
                  ylabel='Phase Error (degrees)',
                  title='(e) AM/PM Characteristic')

    ph_pa  = np.degrees(np.angle(pa_c[idx])  - np.angle(ref_c[idx]))
    ph_dpd = np.degrees(np.angle(dpd_c[idx]) - np.angle(ref_c[idx]))
    ax_ampm.scatter(a_in_n[s][::2], ph_pa[s][::2],  s=1.5, alpha=0.35,
                    color=C_PA,  label='PA (no DPD)', zorder=3)
    ax_ampm.scatter(a_in_n[s][::2], ph_dpd[s][::2], s=1.5, alpha=0.35,
                    color=C_DPD, label=f'{backbone_label}', zorder=4)
    ax_ampm.axhline(0, color=C_REF, lw=0.8, ls='--', zorder=5)
    ax_ampm.legend(fontsize=7.5, frameon=True, edgecolor='black',
                   fancybox=False, markerscale=4)

    # ── (f)(g) Constellations ─────────────────────────────────────────────────
    def plot_const(ax, sig, title, color):
        academic_axes(ax, xlabel='In-Phase (I)', ylabel='Quadrature (Q)', title=title)
        idx2 = np.random.choice(len(sig), min(3000, len(sig)), replace=False)
        ax.scatter(sig[idx2].real, sig[idx2].imag,
                   s=1.5, alpha=0.3, color=color, zorder=3)
        lim = np.percentile(np.abs(sig), 99) * 1.2
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.axhline(0, color='black', lw=0.4, alpha=0.3)
        ax.axvline(0, color='black', lw=0.4, alpha=0.3)

    plot_const(fig.add_subplot(gs[2, 0]),
               pa_c,  '(f) Constellation — PA Output (No DPD)', C_PA)
    plot_const(fig.add_subplot(gs[2, 1]),
               dpd_c, f'(g) Constellation — {backbone_label} Output', C_DPD)

    # ── (h) Summary table ─────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 2])
    ax_tbl.set_facecolor('white')
    for sp in ax_tbl.spines.values():
        sp.set_color('black'); sp.set_linewidth(0.8)
    ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
    ax_tbl.set_title('(h) Performance Summary',
                     fontsize=9, fontweight='bold', pad=5)

    rows = [
        ['EVM (no DPD)',    f'{evm_before:.2f} %'],
        ['EVM (with DPD)',  f'{evm_after:.2f} %'],
        ['EVM reduction',   f'{evm_before - evm_after:+.2f} %'],
        ['NMSE (no DPD)',   f'{nmse_before:.1f} dB'],
        ['NMSE (with DPD)', f'{nmse_after:.1f} dB'],
        ['NMSE gain',       f'{nmse_before - nmse_after:+.1f} dB'],
        ['Backbone',        backbone_label],
        ['Hidden / Memory', f'{args.hidden} / {args.seq_len}'],
        ['Epochs',          str(args.epochs)],
    ]

    tbl = ax_tbl.table(
        cellText  = [[r[1]] for r in rows],
        rowLabels = [r[0] for r in rows],
        colLabels = ['Value'],
        cellLoc   = 'center',
        rowLoc    = 'left',
        loc       = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#aaaaaa')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#ddeeff')
            cell.set_text_props(fontweight='bold')
        elif r in (3, 6):   # separator rows
            cell.set_facecolor('#f0f4f8')
        elif r % 2 == 0:
            cell.set_facecolor('#f8f8f8')
        else:
            cell.set_facecolor('white')

    out_path = str(out_dir / 'dpd_results.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"  Plot saved -> {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args    = get_args()
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Neural DPD  |  {args.backbone}  |  {args.dataset}")
    print(f"{'='*60}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading '{args.dataset}'...")
    data = opendpd.load_dataset(f'datasets/{args.dataset}')
    print(f"  Train: {len(data['X_train']):,}  "
          f"Val: {len(data['X_val']):,}  "
          f"Test: {len(data['X_test']):,}")

    # Indirect learning:
    #   DPD input  = PA output (y)  — distorted signal from amplifier
    #   DPD target = PA input  (X)  — clean signal we wanted to transmit
    #   Model learns the inverse PA mapping: y -> X
    Xtr, Ytr = make_sequences(data['y_train'], data['X_train'], args.seq_len)
    Xva, Yva = make_sequences(data['y_val'],   data['X_val'],   args.seq_len)
    Xte, _   = make_sequences(data['y_test'],  data['X_test'],  args.seq_len)

    train_loader = make_loader(Xtr, Ytr, args.batch, shuffle=True)
    val_loader   = make_loader(Xva, Yva, args.batch, shuffle=False)

    # ── 2. Baseline ───────────────────────────────────────────────────────────
    print("\n[2/5] Baseline metrics (no DPD)...")
    ref_c = to_complex(data['X_test'])[args.seq_len - 1:]   # clean PA input
    pa_c  = to_complex(data['y_test'])[args.seq_len - 1:]   # distorted PA output

    evm_before  = evm_percent(ref_c, pa_c)
    nmse_before = nmse_db(ref_c, pa_c)
    print(f"  EVM  : {evm_before:.2f} %")
    print(f"  NMSE : {nmse_before:.1f} dB")

    # ── 3. Build model ────────────────────────────────────────────────────────
    print(f"\n[3/5] Building {args.backbone}...")
    ModelClass = BACKBONE_MAP[args.backbone]
    model = (ModelClass(input_size=2, hidden_size=args.hidden, seq_len=args.seq_len)
             if args.backbone == 'my_mlp_dpd'
             else ModelClass(input_size=2, hidden_size=args.hidden))
    model = model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print(f"\n[4/5] Training {args.epochs} epochs...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    train_losses, val_losses = [], []
    best_val  = float('inf')
    best_path = out_dir / f'{args.backbone}_best.pth'
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        train_losses.append(tl)
        val_losses.append(vl)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), best_path)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"train={tl:.6f}  val={vl:.6f}  [{time.time()-t0:.0f}s]")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print(f"\n[5/5] Evaluating on test set...")
    model.load_state_dict(torch.load(best_path, map_location=device))

    # dpd_pred_c = what the DPD outputs (the pre-distorted signal to feed the PA)
    # We measure how close this is to the original clean reference signal
    dpd_pred_c = predict_complex(model, Xte, args.batch, device)

    n = min(len(ref_c), len(dpd_pred_c))
    # FIX: evm_after compares DPD output vs clean reference — not pa vs pa
    evm_after  = evm_percent(ref_c[:n], dpd_pred_c[:n])
    nmse_after = nmse_db(ref_c[:n],     dpd_pred_c[:n])

    print(f"\n  +------------------------------------------+")
    print(f"  |           FINAL RESULTS                  |")
    print(f"  +------------------------------------------+")
    print(f"  |  EVM  before DPD : {evm_before:>8.2f} %          |")
    print(f"  |  EVM  after  DPD : {evm_after:>8.2f} %          |")
    print(f"  |  EVM  reduction  : {evm_before-evm_after:>+8.2f} %          |")
    print(f"  |  NMSE before DPD : {nmse_before:>8.1f} dB         |")
    print(f"  |  NMSE after  DPD : {nmse_after:>8.1f} dB         |")
    print(f"  |  NMSE gain       : {nmse_before-nmse_after:>+8.1f} dB         |")
    print(f"  +------------------------------------------+")

    print("\n  Generating plots...")
    make_plots(
        args,
        train_losses, val_losses,
        ref_c  = ref_c[:n],
        pa_c   = pa_c[:n],
        dpd_c  = dpd_pred_c[:n],
        evm_before=evm_before,   evm_after=evm_after,
        nmse_before=nmse_before, nmse_after=nmse_after,
        out_dir=out_dir,
    )

    torch.save(model.state_dict(), out_dir / f'{args.backbone}_final.pth')
    print(f"\n  Done. All outputs in: {out_dir}/\n")


if __name__ == '__main__':
    main()
