import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from numpy.fft import rfft, irfft, rfftfreq

from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# ------------------------------- Data classes -------------------------------
@dataclass(frozen=True)
class RFModel:
    model_state: dict
    x_mean: float
    x_std: float
    v_mean: float
    v_std: float
    m_eff: float
    k0: float
    c0: float

def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)

# ------------------------------- Signal helpers -------------------------------

def bandpass_filter(x: np.ndarray,
                    f_lo: float,
                    f_hi: float,
                    fs: float,
                    order: int = 4) -> np.ndarray:
    """"
    Butterworth bandpass filter.
    
    Inputs:
    - x: input signal
    - f_lo: low cutoff frequency (Hz)
    - f_hi: high cutoff frequency (Hz)
    - fs: sampling frequency (Hz)
    - order: filter order
    
    Ouptuts:
    - y: bandpassed signal
    """ 
    nyquist = 0.5 * fs
    low = f_lo / nyquist
    high = f_hi / nyquist
    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, x)


def band_limited_integrate_fft(a_t: np.ndarray,
                               fs: float,
                               f_lo: float, 
                               f_hi: float,
                               order: int) -> np.ndarray:
    """
    Integrate in frequency domain with band-limiting.
    
    Inputs:
    - a_t: time-domain acceleration signal
    - fs: sampling frequency (Hz)
    - f_lo: low cutoff frequency (Hz)
    - f_hi: high cutoff frequency (Hz)
    - order: integration order (1 for velocity, 2 for displacement)
    
    Outputs:
    - integrated signal in time domain
    """
    if order < 0:
        raise ValueError("order must be >= 0")
    n = len(a_t)
    A = rfft(a_t - np.mean(a_t))
    f = rfftfreq(n, 1.0 / fs)

    mask = (f >= f_lo) & (f <= f_hi)
    Y = np.zeros_like(A, dtype=np.complex128)

    if order == 0:
        Y[mask] = A[mask]
    else:
        omega = 2.0 * np.pi * f
        denom = (1j * omega) ** order
        denom[0] = np.inf  # avoid DC blow-up
        Y[mask] = A[mask] / denom[mask]

    return irfft(Y, n=n)


def compute_relative_motion(accel_i: np.ndarray,
                            accel_j: np.ndarray,
                            fs: float,
                            f_lo: float,
                            f_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative displacement and velocity from accelerations.
    
    Inputs:
    - accel_i: acceleration at location i
    - accel_j: acceleration at location j
    - fs: sampling frequency (Hz)
    - f_lo: low cutoff frequency (Hz)
    - f_hi: high cutoff frequency (Hz)
    
    Outputs:
    - x_rel: relative displacement (j - i)
    - v_rel: relative velocity (j - i)
    """
    
    a_rel = accel_j - accel_i
    v_rel = band_limited_integrate_fft(a_rel, fs, f_lo, f_hi, order=1)
    x_rel = band_limited_integrate_fft(a_rel, fs, f_lo, f_hi, order=2)
    return x_rel, v_rel

# ------------------------------- Neural Network Model -------------------------------

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from architecture import RFMLP

def train_neural_network(x_train: np.ndarray,
                         v_train: np.ndarray,
                         rf_train: np.ndarray,
                         batch_size: int = 1024,
                         epochs: int = 100,
                         learning_rate: float = 1e-3,
                         device: str = 'cpu') -> Tuple[dict, float, float, float, float]:
    """
    Train neural network to predict restoring force from relative motion.
    
    Inputs:
    - x_train: relative displacement array
    - v_train: relative velocity array
    - rf_train: restoring force array
    - batch_size: batch size for training
    - epochs: number of training epochs
    - learning_rate: learning rate for Adam optimizer
    - device: 'cpu' or 'cuda'
    
    Outputs:
    - model_state: trained model state_dict
    - x_mean, x_std, v_mean, v_std: normalization statistics
    """
    # Compute normalization statistics
    x_mean, x_std = float(x_train.mean()), float(x_train.std())
    v_mean, v_std = float(v_train.mean()), float(v_train.std())
    
    # Normalize inputs
    x_norm = (x_train - x_mean) / x_std
    v_norm = (v_train - v_mean) / v_std
    
    # Create tensors
    X = torch.tensor(np.stack([x_norm, v_norm], axis=1), dtype=torch.float32)
    y = torch.tensor(rf_train.reshape(-1, 1), dtype=torch.float32)
    
    # Create DataLoader & minibatching
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = RFMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    print(f"Training neural network for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_X)
        
        epoch_loss /= len(dataset)
        if (epoch + 1) % 1 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6e}")
    
    print(f"Training complete!")
    return model.state_dict(), x_mean, x_std, v_mean, v_std


def evaluate_nn_rf(x_rel: np.ndarray,
                   v_rel: np.ndarray,
                   model_state: dict,
                   x_mean: float,
                   x_std: float,
                   v_mean: float,
                   v_std: float,
                   device: str = 'cpu') -> np.ndarray:
    """
    Evaluate neural network RF model at given (x_rel, v_rel) points.
    
    Inputs:
    - x_rel: relative displacement array
    - v_rel: relative velocity array
    - model_state: trained model state_dict
    - x_mean, x_std, v_mean, v_std: normalization statistics
    - device: 'cpu' or 'cuda'
    
    Outputs:
    - rf_eval: evaluated restoring force array
    """
    # Normalize inputs
    x_norm = (x_rel - x_mean) / x_std
    v_norm = (v_rel - v_mean) / v_std
    
    # Create tensor
    X = torch.tensor(np.stack([x_norm, v_norm], axis=1), dtype=torch.float32).to(device)
    
    # Load model and evaluate
    model = RFMLP().to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    with torch.no_grad():
        rf_pred = model(X).cpu().numpy().flatten()
    
    return rf_pred


# ------------------------------- Simulation -------------------------------
def simulate_ode(x2: np.ndarray,
                 v2: np.ndarray,
                 x3_measured: np.ndarray,
                 v3_measured: np.ndarray,
                 fs: float,
                 rf_model: RFModel,
                 n_init: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate acceleration at location 3 using trained NN RF model.
    NN is evaluated via NumPy (no Torch in RHS).
    """

    # ---------------- Setup ----------------
    N = len(x2)
    t = np.arange(N) / fs

    x3_full = np.zeros(N)
    v3_full = np.zeros(N)
    a3_full = np.zeros(N)

    # Initialization (measured)
    x3_full[:n_init] = x3_measured[:n_init]
    v3_full[:n_init] = v3_measured[:n_init]

    # ---------------- Load NN once ----------------
    from architecture import RFMLP

    rf_nn = RFMLP()
    rf_nn.load_state_dict(rf_model.model_state)
    rf_nn.eval()

    # Extract weights (NumPy)
    W1 = rf_nn.fc1.weight.detach().numpy()
    b1 = rf_nn.fc1.bias.detach().numpy()
    W2 = rf_nn.fc2.weight.detach().numpy()
    b2 = rf_nn.fc2.bias.detach().numpy()
    W3 = rf_nn.fc3.weight.detach().numpy()
    b3 = rf_nn.fc3.bias.detach().numpy()

    # ---------------- NumPy RF evaluator ----------------
    def rf_eval_scalar(x_rel, v_rel):
        # Linear part
        rf_lin = rf_model.k0 * x_rel + rf_model.c0 * v_rel

        # NN residual
        x = np.array([
            (x_rel - rf_model.x_mean) / rf_model.x_std,
            (v_rel - rf_model.v_mean) / rf_model.v_std
        ])

        z1 = np.tanh(W1 @ x + b1)
        z2 = np.tanh(W2 @ z1 + b2)
        rf_nl = (W3 @ z2 + b3)[0]
        
        return rf_lin + rf_nl


    # Initial acceleration values
    for i in range(n_init):
        x_rel = x3_full[i] - x2[i]
        v_rel = v3_full[i] - v2[i]
        a3_full[i] = rf_eval_scalar(x_rel, v_rel) / rf_model.m_eff

    # ---------------- Interpolators ----------------
    x2_fun = interp1d(t, x2, kind="cubic", fill_value="extrapolate")
    v2_fun = interp1d(t, v2, kind="cubic", fill_value="extrapolate")

    # ---------------- ODE RHS ----------------
    def rhs(tau, state):
        x3, v3 = state
        x_rel = x3 - x2_fun(tau)
        v_rel = v3 - v2_fun(tau)
        a3 = rf_eval_scalar(x_rel, v_rel) / rf_model.m_eff
        return [v3, a3]

    # ---------------- Solve ODE ----------------
    sol = solve_ivp(
        fun=rhs,
        t_span=(t[n_init - 1], t[-1]),
        y0=[x3_measured[n_init - 1], v3_measured[n_init - 1]],
        t_eval=t[n_init:],
        method="RK23",
        rtol=1e-5,
        atol=1e-7
    )

    if not sol.success:
        print(f"WARNING: ODE solver failed: {sol.message}")

    # Store solution
    x3_full[n_init:] = sol.y[0]
    v3_full[n_init:] = sol.y[1]

    # Compute accelerations
    for i in range(n_init, N):
        x_rel = x3_full[i] - x2[i]
        v_rel = v3_full[i] - v2[i]
        a3_full[i] = rf_eval_scalar(x_rel, v_rel) / rf_model.m_eff

    return t, a3_full
