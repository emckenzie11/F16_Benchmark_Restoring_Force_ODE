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
    band_limited_integrate_fft,
    compute_relative_motion,
    train_neural_network,
    simulate_ode
)


@dataclass(frozen=True)
class Config:
    # Data parameters
    fs: float = 400.0
    location_i: int = 2  # measured accel location
    location_j: int = 3  # across nonlinear connection

    # Identification parameters
    m2: float = 1.0
    m3: float = 1.0
    
    # Neural network training parameters
    batch_size: int = 1024
    epochs: int = 20 
    learning_rate: float = 3e-3

    # Mode isolation parameters
    freq_low: float = 6.8
    freq_high: float = 8.6


def main() -> None:
    seed_everything(0)
    cfg = Config()

    # ------------------------------- Process Data ------------------------------- 
    train_val_y1, test_y1 = nonlinear_benchmarks.F16(output_index=0)  # location 1
    train_val_y2, test_y2 = nonlinear_benchmarks.F16(output_index=1)  # location 2
    train_val_y3, test_y3 = nonlinear_benchmarks.F16(output_index=2)  # location 3

    x_rel_all: List[np.ndarray] = []
    v_rel_all: List[np.ndarray] = []
    rf_residual_all: List[np.ndarray] = []

    # Storage for linear identification
    A_lin_all = []
    b_lin_all = [] 

    # -------------------- Linear identification --------------------
    A32_all, b32_all = [], []   # for eq (3)
    A21_all, b21_all = [], []   # for eq (2)
    y2_all  = []                # LHS for eq (2)

    for level in (0, 4):

        a1 = train_val_y1[level].y.flatten()
        a2 = train_val_y2[level].y.flatten()
        a3 = train_val_y3[level].y.flatten()

        # Relative motions
        x32, v32 = compute_relative_motion(a2, a3, cfg.fs, cfg.freq_low, cfg.freq_high)
        x21, v21 = compute_relative_motion(a1, a2, cfg.fs, cfg.freq_low, cfg.freq_high)

        # Bandpassed accelerations
        a3_bp = bandpass_filter(a3, cfg.freq_low, cfg.freq_high, cfg.fs)
        a2_bp = bandpass_filter(a2, cfg.freq_low, cfg.freq_high, cfg.fs)

        # ---------------- Eq (3): identify k3, c3 ----------------
        A32_all.append(np.column_stack([x32, v32]))
        b32_all.append(-cfg.m3 * a3_bp)

        # ---------------- Eq (2): build LHS ----------------
        A21_all.append(np.column_stack([x21, v21]))
        y2_all.append(cfg.m2 * a2_bp)

    # ---------------- Solve for k3, c3 ----------------
    A32 = np.vstack(A32_all)
    b32 = np.concatenate(b32_all)

    k3, c3 = np.linalg.lstsq(A32, b32, rcond=None)[0]

    print(f"k3 = {k3:.6e}")
    print(f"c3 = {c3:.6e}")
    
    # ---------------- Solve for k2, c2 ----------------
    y2 = np.concatenate(y2_all)
    x32_all = A32[:, 0]
    v32_all = A32[:, 1]

    y2_eff = y2 - k3 * x32_all - c3 * v32_all

    A21 = np.vstack(A21_all)

    k2, c2 = np.linalg.lstsq(A21, y2_eff, rcond=None)[0]

    print(f"k2 = {k2:.6e}")
    print(f"c2 = {c2:.6e}")

    # -------- Build residual training data (other levels) --------
    for level in range(0, 8):
        if level in (0, 4):
            continue

        a2_train = train_val_y2[level].y.flatten()
        a3_train = train_val_y3[level].y.flatten()

        # Relative motion
        x_rel, v_rel = compute_relative_motion(a2_train, a3_train, cfg.fs, cfg.freq_low, cfg.freq_high)

        # Restoring force
        a3_train_bp = bandpass_filter(a3_train, cfg.freq_low, cfg.freq_high, cfg.fs)
        rf = -cfg.m3 * a3_train_bp

        # Residual
        rf_residual = rf - (k3 * x_rel + c3 * v_rel)

        x_rel_all.append(x_rel)
        v_rel_all.append(v_rel)
        rf_residual_all.append(rf_residual)

    # ---------------------------- Concatenate All Levels ----------------------------
    x_train = np.concatenate(x_rel_all)
    v_train = np.concatenate(v_rel_all)
    rf_train = np.concatenate(rf_residual_all)

    # ----------------------------------- Training -----------------------------------
    print("-" * 79)
    print("Training Neural Network Restoring Force Residual Model")
    print("-" * 79)
    print(f"Training samples:  {len(x_train)}")
    print(f"Batch size:        {cfg.batch_size}")
    print(f"Epochs:            {cfg.epochs}")
    print("-" * 79)

    model_state, x_mean, x_std, v_mean, v_std = train_neural_network(
        x_train=x_train,
        v_train=v_train,
        rf_train=rf_train,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        device="cpu",)

    # --------------------------- Create Restoring Force Model ---------------------------
    rf_model = RFModel(
        model_state=model_state,
        x_mean=x_mean,
        x_std=x_std,
        v_mean=v_mean,
        v_std=v_std,
        m_eff=cfg.m3,
        k0=k3,
        c0=c3)

    print("-" * 79)
    print("Neural Network Model Created")
    print("-" * 79)
    
    # --------------------------- Simulation and Testing ---------------------------
    results = {}
    for test_idx in range(0, 6):
        print(f"Testing ODE simulation on test index {test_idx}...")
        
        # Get measured data
        a2_test = test_y2[test_idx].y.flatten()
        a3_test = test_y3[test_idx].y.flatten()
        
        # Bandpass filter a3_test for error computation
        a3_test_bp = bandpass_filter(a3_test, cfg.freq_low, cfg.freq_high, cfg.fs)
        
        # Compute measured states at location 2 (driving input)
        x2_test = band_limited_integrate_fft(a2_test, cfg.fs, cfg.freq_low, cfg.freq_high, order=2)
        v2_test = band_limited_integrate_fft(a2_test, cfg.fs, cfg.freq_low, cfg.freq_high, order=1)
        
        # Compute measured states at location 3 (for initial conditions)
        x3_test = band_limited_integrate_fft(a3_test, cfg.fs, cfg.freq_low, cfg.freq_high, order=2)
        v3_test = band_limited_integrate_fft(a3_test, cfg.fs, cfg.freq_low, cfg.freq_high, order=1)
        
        # Simulate acceleration at location 3
        t_sim, a3_sim = simulate_ode(
            x2=x2_test,
            v2=v2_test,
            x3_measured=x3_test,
            v3_measured=v3_test,
            fs=cfg.fs,
            rf_model=rf_model,
            n_init=100
        )
        
        # Truncate bandpassed test signal for error computation
        a3_test_bp_trunc = a3_test_bp[:len(a3_sim)]
    
        # Compute RMSE
        accel_3_error = np.sqrt(
            np.mean((a3_sim - a3_test_bp_trunc) ** 2)
        )
        
        # Store results
        results[test_idx] = {
            't_sim': t_sim,
            'a3_sim': a3_sim,
            'a3_test_bp': a3_test_bp,
            'rmse': accel_3_error
        }

        excitation = "Multi Sine" if test_idx < 3 else "Sine Sweep"
        level_label = ["Level 2", "Level 4", "Level 6"][test_idx % 3]

        print(f"Simulation completed for {excitation} {level_label} (idx {test_idx})")
        print(f"  RMSE a3: {accel_3_error:.6f}")
        print("-" * 79)
        
    # ----------------------------------- Plotting -----------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 6), sharey=True)

    level_labels = ['Level 2', 'Level 4', 'Level 6']
    row_labels = ['Multi Sine', 'Sine Sweep']

    # Row-wise max time for proper spanning
    tmax_row = {0: 0.0, 1: 0.0}
    for idx, test_idx in enumerate(range(6)):
        row = 0 if idx < 3 else 1
        t_sim = results[test_idx]['t_sim']
        tmax_row[row] = max(tmax_row[row], float(t_sim[-1]))

    for idx, test_idx in enumerate(range(6)):
        row = 0 if idx < 3 else 1
        col = idx % 3
        ax = axes[row, col]

        t_sim   = results[test_idx]['t_sim']
        a3_sim  = results[test_idx]['a3_sim']
        a3_test = results[test_idx]['a3_test_bp'][:len(a3_sim)]
        rmse    = results[test_idx]['rmse']

        ax.plot(t_sim, a3_test, label='Measured (bandpassed)', color='tab:blue')
        ax.plot(t_sim, a3_sim, '--', label='Simulated', color='tab:red')

        ax.set_xlim(0.0, tmax_row[row])

        if row == 0:
            ax.set_title(level_labels[col], fontsize=14)

        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=14)

        if row == 1:
            ax.set_xlabel('Time [s]')

        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.95, f"RMSE={rmse:.3e}", transform=ax.transAxes,
                va='top', ha='left', fontsize=9)

        if col == 2:
            ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
