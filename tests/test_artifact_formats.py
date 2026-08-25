from __future__ import annotations

from pathlib import Path

import torch

from flowsis.utils.training import load_training_state, save_training_state


def test_training_state_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    save_training_state(
        tmp_path,
        optimizer,
        scheduler,
        epoch=3,
        global_step=17,
    )
    state = load_training_state(
        tmp_path,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert state.epoch == 3
    assert state.global_step == 17
