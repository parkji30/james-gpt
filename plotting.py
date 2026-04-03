
from pathlib import Path
import time

import matplotlib.pyplot as plt

class LossPlotter:
    def __init__(
        self,
        update_every=25,
        max_points=1000,
        output_path="training_loss.png",
        min_refresh_interval=0.5,
        save_interval=5.0,
        raw_window_points=1000,
        interactive=False,
    ):
        self.update_every = update_every
        self.max_points = max_points
        self.output_path = Path(output_path)
        self.min_refresh_interval = min_refresh_interval
        self.save_interval = save_interval
        self.raw_window_points = raw_window_points
        self.interactive = interactive
        self.steps = []
        self.losses = []
        self.smoothed_losses = []
        self.ema_decay = 0.97
        self.last_render_time = 0.0
        self.last_save_time = 0.0

        if self.interactive:
            plt.ion()
        self.figure, self.axis = plt.subplots(figsize=(8, 4.5))
        self.raw_line, = self.axis.plot(
            [],
            [],
            color="tab:blue",
            alpha=0.95,
            linewidth=1.8,
            label="loss",
            zorder=2,
        )
        self.smooth_line, = self.axis.plot(
            [],
            [],
            color="tab:red",
            alpha=0.8,
            linewidth=1.8,
            linestyle="--",
            label="ema",
            zorder=1,
        )
        self.axis.set_title("Training Loss")
        self.axis.set_xlabel("Step")
        self.axis.set_ylabel("Loss")
        self.axis.grid(alpha=0.2)
        self.axis.legend()
        self.figure.tight_layout()

    def _downsample(self, xs, ys, max_points):
        if len(xs) <= max_points:
            return xs, ys

        stride = max(1, len(xs) // max_points)
        sampled_xs = xs[::stride]
        sampled_ys = ys[::stride]

        if sampled_xs[-1] != xs[-1]:
            sampled_xs = sampled_xs + [xs[-1]]
            sampled_ys = sampled_ys + [ys[-1]]

        return sampled_xs, sampled_ys

    def _render(self, force_save=False):
        ema_steps, ema_losses = self._downsample(
            self.steps,
            self.smoothed_losses,
            self.max_points,
        )
        raw_steps = self.steps[-self.raw_window_points:]
        raw_losses = self.losses[-self.raw_window_points:]
        raw_max_points = max(200, self.max_points // 2)
        raw_steps, raw_losses = self._downsample(raw_steps, raw_losses, raw_max_points)

        self.raw_line.set_data(raw_steps, raw_losses)
        self.smooth_line.set_data(ema_steps, ema_losses)
        self.axis.relim()
        self.axis.autoscale_view()

        if self.interactive:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(0.001)

        now = time.monotonic()
        if force_save or now - self.last_save_time >= self.save_interval:
            self.figure.savefig(self.output_path, dpi=120)
            self.last_save_time = now

    def update(self, step, loss):
        self.steps.append(step)
        self.losses.append(loss)

        if self.smoothed_losses:
            smoothed_loss = self.ema_decay * self.smoothed_losses[-1] + (1 - self.ema_decay) * loss
        else:
            smoothed_loss = loss
        self.smoothed_losses.append(smoothed_loss)

        if step % self.update_every != 0:
            return

        now = time.monotonic()
        if now - self.last_render_time < self.min_refresh_interval:
            return

        self._render()
        self.last_render_time = now

    def close(self):
        if self.steps:
            self._render(force_save=True)
        if self.interactive:
            plt.ioff()
        plt.close(self.figure)
