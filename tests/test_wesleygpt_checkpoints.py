# Wesley wrote this
"""When SFT writes a checkpoint."""
from wesleygpt.checkpoints import checkpoint_due


def test_always_saves_the_last_step():
    assert checkpoint_due(step=1321, last_step=True, every=200)


def test_saves_every_n_steps_mid_run():
    assert [s for s in range(1, 1000) if checkpoint_due(s, False, 200)] == [200, 400, 600, 800]


def test_no_save_before_training_starts():
    assert not checkpoint_due(step=0, last_step=False, every=200)


def test_disabled_periodic_saves_still_save_the_last_step():
    assert not checkpoint_due(step=200, last_step=False, every=-1)
    assert checkpoint_due(step=1321, last_step=True, every=-1)
