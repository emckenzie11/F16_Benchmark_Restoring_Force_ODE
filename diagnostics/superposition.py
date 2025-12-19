from dataclasses import dataclass
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import nonlinear_benchmarks
import matplotlib.pyplot as plt

from helpers import (
    RFModel,
    seed_everything,
    bandpass_filter,
    #band_limited_integrate_fft,
    compute_relative_motion,
    #train_neural_network,
    #simulate_ode
)


@dataclass(frozen=True)
class Config:
    fs: float = 400.0
    m_eff: float = 1.0

def main() -> None:
    seed_everything(0)
    cfg = Config()

    #--------------------- DIAGNOSTIC SETUP ------------------------
    bandpasses = [
        (3.1, 3.6),   # rigid-body modes
        (3.6, 4.5),   # rigid-body modes
        (4.7, 5.6),   # first flexible mode
        (5.8, 6.7),   # second flexible mode
        (6.8, 8.6),   # second flexible mode
        (8.8, 15.0)]   # third flexible mode 
    
    #--------------------- LOAD DATA ONCE ------------------------
    train_val_y2, _ = nonlinear_benchmarks.F16(output_index=1)
    train_val_y3, _ = nonlinear_benchmarks.F16(output_index=2)
    data_store = {}    

    #--------------------- LOOP OVER BANDPASSES ------------------------
    for freq_low, freq_high in bandpasses:

        print("-" * 79)
        print(f"Diagnostics for bandpass [{freq_low:.1f}, {freq_high:.1f}] Hz")

        data_store[(freq_low, freq_high)] = {}

        # Linear parameter identification (Levels 0 & 4 only)
        A_lin_all, b_lin_all = [], []

        for level in (0, 4):
            a2 = train_val_y2[level].y.flatten()
            a3 = train_val_y3[level].y.flatten()

            x_rel, v_rel = compute_relative_motion(a2, a3, cfg.fs, freq_low, freq_high)

            a3_bp = bandpass_filter(a3, freq_low, freq_high, cfg.fs)
            rf = -cfg.m_eff * a3_bp

            A_lin_all.append(np.column_stack([x_rel, v_rel]))
            b_lin_all.append(rf)

        A_lin = np.vstack(A_lin_all)
        b_lin = np.concatenate(b_lin_all)

        k0, c0 = np.linalg.lstsq(A_lin, b_lin, rcond=None)[0]

        print(f"Linear parameters: k0 = {k0:.3e}, c0 = {c0:.3e}")

        # ------------------------------------------------------------------
        # Build restoring-force diagnostics (ALL levels, stored per-level)
        # ------------------------------------------------------------------
        for level in range(len(train_val_y2)):

            a2 = train_val_y2[level].y.flatten()
            a3 = train_val_y3[level].y.flatten()

            x_rel, v_rel = compute_relative_motion(a2, a3, cfg.fs, freq_low, freq_high)

            a3_bp = bandpass_filter(a3, freq_low, freq_high, cfg.fs)
            rf = -cfg.m_eff * a3_bp

            data_store[(freq_low, freq_high)][level] = {
                "x": x_rel,
                "v": v_rel,
                "rf": rf,
            }
            
    print("-" * 79)
    
    # ----------------------------- PLOTS All Levels ------------------------------
    signal = train_val_y2
    n_levels = 8  # total levels to plot

    fig, axes = plt.subplots(4, 2, figsize=(16, 10))
    
    band_colors = {
    (3.1, 3.6): "tab:cyan",    # rigid-body (low)
    (3.6, 4.5): "tab:purple",  # rigid-body (high)
    (4.7, 5.6): "tab:blue",    # 1st flexible mode (~5.2 Hz)
    (5.8, 6.7): "tab:orange",  # 2nd flexible (low side)
    (6.8, 8.6): "tab:green",   # 2nd flexible (main)
    (8.8, 15.0): "tab:red"}     # 3rd flexible (~9.2 Hz)

    for level in (range(n_levels)):

        row = level % 4
        col = level // 4

        ax = axes[row, col]

        a_full = signal[level].y.flatten()
        t = np.arange(len(a_full)) / cfg.fs

        # Full unfiltered signal
        ax.plot(t, a_full, color="k", lw=1.0, alpha=0.6, label="Full $a_3$")

        # Band-passed overlays
        for freq_low, freq_high in bandpasses:
            a2_bp = bandpass_filter(a_full, freq_low, freq_high, cfg.fs)
            ax.plot(t, a2_bp, lw=1.2, color=band_colors[(freq_low, freq_high)], label=f"{freq_low:.1f}–{freq_high:.1f} Hz")
            
        ax.set_ylabel(f"Level {level}")
        ax.grid(True, alpha=0.3)

    # Labels and legend
    axes[3, 0].set_xlabel("Time [s]")
    axes[3, 1].set_xlabel("Time [s]")
    axes[0, 1].legend(loc = "upper right", ncol=2, fontsize=9)

    fig.suptitle("Measured $a_2$ — Full vs Band-passed Signals (All Levels)", fontsize=14, y=0.995)

    plt.tight_layout()
    plt.show()
    
    # ----------------------------- PLOTS Single Level ------------------------------
    signal = train_val_y2
    test_idx = 4  # Sine Sweep, Level 1

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))

    a_full = signal[test_idx].y.flatten()
    t = np.arange(len(a_full)) / cfg.fs

    # Full unfiltered signal
    ax.plot(t, a_full, color="k", lw=1.0, alpha=0.6, label="Full $a_2$")

    # Band-passed overlays
    for freq_low, freq_high in bandpasses:
        a_bp = bandpass_filter(a_full, freq_low, freq_high, cfg.fs)
        ax.plot(t, a_bp, lw=1.2, color=band_colors[(freq_low, freq_high)], label=f"{freq_low:.1f}–{freq_high:.1f} Hz")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration $a_2$")
    ax.set_title("Measured $a_2$ — Full vs Band-passed (Sine Sweep, Level 1)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=9)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()

