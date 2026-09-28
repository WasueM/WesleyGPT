# Wesley wrote this
"""When SFT writes a checkpoint, and how much memory the training process holds."""
import psutil


def checkpoint_due(step, last_step, every):
    """True at the last step, and every `every` steps before it (every <= 0 disables
    the periodic saves). The periodic ones mean a crash late in a seven-hour run
    still leaves a nearly-finished model on disk."""
    return last_step or (every > 0 and step > 0 and step % every == 0)


def rss_gb():
    """Resident memory of this process in GB, logged each step so a leak shows as a trend."""
    return psutil.Process().memory_info().rss / 1e9
