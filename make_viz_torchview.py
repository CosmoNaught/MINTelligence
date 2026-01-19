"""
Visualize the 4 ESA models (GRU/LSTM × prevalence/cases) as PNGs.

Run:
    python make_viz_torchview.py

Requires:
    pip install torch torchview graphviz

Make sure graphviz is installed on the system so .render() can write PNGs.
"""

import torch
import torch.nn as nn
from torchview import draw_graph


# ===== model defs matching your training script =====

class GRUModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_prob: float,
        num_layers: int = 1,
        predictor: str = "prevalence",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predictor = predictor

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, features)
        out, _ = self.gru(x)
        out = self.ln(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.activation(out)
        return out


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_prob: float,
        num_layers: int = 1,
        predictor: str = "prevalence",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predictor = predictor

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, features)
        out, _ = self.lstm(x)
        out = self.ln(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.activation(out)
        return out


# ===== this matches your ESA-ish input shape =====
# 2 (sin/cos) + 12 static + 2 dynamic = 16
INPUT_SIZE = 16
OUTPUT_SIZE = 1

MODEL_SPECS = {
    "lstm_prevalence": dict(
        cls=LSTMModel,
        hidden_size=128,
        num_layers=2,
        dropout=0.10,
        lookback=78,
        predictor="prevalence",
    ),
    "lstm_cases": dict(
        cls=LSTMModel,
        hidden_size=256,
        num_layers=2,
        dropout=0.425,
        lookback=26,
        predictor="cases",
    ),
    "gru_prevalence": dict(
        cls=GRUModel,
        hidden_size=256,
        num_layers=2,
        dropout=0.352,
        lookback=78,
        predictor="prevalence",
    ),
    "gru_cases": dict(
        cls=GRUModel,
        hidden_size=512,
        num_layers=2,
        dropout=0.212,
        lookback=13,
        predictor="cases",
    ),
}


def main():
    for name, spec in MODEL_SPECS.items():
        print(f"rendering {name} ...")

        model = spec["cls"](
            input_size=INPUT_SIZE,
            hidden_size=spec["hidden_size"],
            output_size=OUTPUT_SIZE,
            dropout_prob=spec["dropout"],
            num_layers=spec["num_layers"],
            predictor=spec["predictor"],
        )
        model.eval()

        dummy_input = torch.randn(spec["lookback"], 1, INPUT_SIZE)

        # draw graph; depth>1 so we don't get the single-box view
        graph = draw_graph(
            model,
            input_data=dummy_input,
            expand_nested=True,
            depth=3,
        )

        # your version of torchview returns a ComputationGraph with .visual_graph
        # which is a graphviz.Digraph
        try:
            gv = graph.visual_graph  # graphviz.Digraph
            gv.render(filename=name, format="png", cleanup=True)
        except AttributeError:
            # in case someone upgrades later
            graph.visualize(filename=f"{name}.png", format="png")

    print("done.")


if __name__ == "__main__":
    main()
