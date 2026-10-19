"""Small four-qubit hybrid classifiers inspired by Sebastianelli et al."""
from __future__ import annotations

import math
import torch
from torch import nn

from config import FEATURES

N_QUBITS = 4


class Conv1dEncoder(nn.Module):
    output_dim = 64

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(len(FEATURES), 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))


class Conv2dEncoder(nn.Module):
    output_dim = 64

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class FeatureEncoder(nn.Module):
    def __init__(self, input_mode: str, gaf_mode: str):
        super().__init__()
        self.input_mode = input_mode
        channels = len(FEATURES) * (2 if gaf_mode == "both" else 1)
        self.one_d = Conv1dEncoder() if input_mode in ("conv1d", "fusion") else None
        self.two_d = Conv2dEncoder(channels) if input_mode in ("conv2d", "fusion") else None
        raw_dim = 128 if input_mode == "fusion" else 64
        self.fuse = nn.Sequential(nn.Linear(raw_dim, 64), nn.ReLU()) if input_mode == "fusion" else nn.Identity()

    def forward(self, batch):
        parts = []
        if self.one_d is not None:
            parts.append(self.one_d(batch["x1d"]))
        if self.two_d is not None:
            parts.append(self.two_d(batch["x2d"]))
        return self.fuse(torch.cat(parts, dim=1) if len(parts) == 2 else parts[0])


class RealAmplitudesLayer(nn.Module):
    """Adjoint ideal training plus optional noisy finite-shot evaluation.

    Adjoint differentiation is simulator-only and requires a unitary circuit.
    Noise is therefore enabled only in eval mode, which mirrors deploying a
    classically trained checkpoint for forward execution on quantum hardware.
    """
    def __init__(self, differentiator: str, noise=None):
        super().__init__()
        if differentiator != "adjoint":
            raise ValueError("This experiment intentionally supports adjoint training only")
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError("Install requirements-noise.txt for adjoint/noisy PennyLane execution") from exc
        self.qml = qml
        self.noise = noise
        self.weight = nn.Parameter(torch.empty(2, N_QUBITS).uniform_(-0.1, 0.1))
        ideal_device = qml.device("default.qubit", wires=N_QUBITS, shots=None)
        noisy_device = qml.device("default.mixed", wires=N_QUBITS, shots=noise["shots"]) if noise else None

        @qml.qnode(ideal_device, interface="torch", diff_method="adjoint")
        def ideal_circuit(inputs, weights):
            for wire in range(N_QUBITS):
                qml.RY(inputs[wire], wires=wire)
            for layer in range(2):
                for wire in range(N_QUBITS):
                    qml.RY(weights[layer, wire], wires=wire)
                for wire in range(N_QUBITS):
                    qml.CNOT(wires=[wire, (wire + 1) % N_QUBITS])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(N_QUBITS)]

        self.ideal_circuit = ideal_circuit
        if noise:
            @qml.qnode(noisy_device, interface="torch", diff_method=None)
            def noisy_circuit(inputs, weights):
                for wire in range(N_QUBITS):
                    qml.RY(inputs[wire], wires=wire)
                    qml.DepolarizingChannel(noise["p1"], wires=wire)
                for layer in range(2):
                    for wire in range(N_QUBITS):
                        qml.RY(weights[layer, wire], wires=wire)
                        qml.DepolarizingChannel(noise["p1"], wires=wire)
                    for wire in range(N_QUBITS):
                        target = (wire + 1) % N_QUBITS
                        qml.CNOT(wires=[wire, target])
                        # Local depolarization on both participating qubits is a
                        # portable approximation of a two-qubit gate error.
                        qml.DepolarizingChannel(noise["p2"] / 2.0, wires=wire)
                        qml.DepolarizingChannel(noise["p2"] / 2.0, wires=target)
                for wire in range(N_QUBITS):
                    qml.BitFlip(noise["readout"], wires=wire)
                return [qml.expval(qml.PauliZ(wire)) for wire in range(N_QUBITS)]
            self.noisy_circuit = noisy_circuit

    def forward(self, angles):
        circuit = self.noisy_circuit if (not self.training and self.noise) else self.ideal_circuit
        # Explicit per-sample execution is accepted by real-device plugins and
        # avoids relying on simulator-only parameter broadcasting.
        return torch.stack([torch.stack(circuit(row, self.weight)).to(torch.float32) for row in angles])


class HybridClassifier(nn.Module):
    """CNN -> classical FC -> 4-qubit layer -> classical FC."""
    def __init__(self, input_mode: str, gaf_mode: str, differentiator: str, noise=None):
        super().__init__()
        self.encoder = FeatureEncoder(input_mode, gaf_mode)
        self.fc_in = nn.Linear(64, N_QUBITS)
        self.quantum = RealAmplitudesLayer(differentiator, noise=noise)
        self.fc_out = nn.Linear(N_QUBITS, 2)

    def forward(self, batch):
        angles = math.pi * torch.tanh(self.fc_in(self.encoder(batch)))
        return self.fc_out(self.quantum(angles))


class MatchedClassicalLayer(nn.Module):
    """Eight-parameter elementwise classical replacement for the eight-angle VQC."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(N_QUBITS))
        self.bias = nn.Parameter(torch.zeros(N_QUBITS))

    def forward(self, values):
        return torch.tanh(values * self.scale + self.bias)


class CNNBaseline(nn.Module):
    """Same CNN and 4-D bottleneck, replacing the VQC by a classical hidden layer."""
    def __init__(self, input_mode: str, gaf_mode: str):
        super().__init__()
        self.encoder = FeatureEncoder(input_mode, gaf_mode)
        self.fc_in = nn.Linear(64, N_QUBITS)
        self.classical_hidden = MatchedClassicalLayer()
        self.fc_out = nn.Linear(N_QUBITS, 2)

    def forward(self, batch):
        z = math.pi * torch.tanh(self.fc_in(self.encoder(batch)))
        return self.fc_out(self.classical_hidden(z))
