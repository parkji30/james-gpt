
from pathlib import Path
import matplotlib.pyplot as plt

class LossPlotter:
    def __init__(self, update_every=25, max_points=1000, output_path="training_loss.png"):
        self.update_every = update_every
        self.output_path = Path(output_path)
        self.steps = []
        self.losses = []
        self.smoothed_losses = []
        self.ema_decay = 0.97

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
            color="tab:orange",
            alpha=0.35,
            linewidth=1.4,
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

        self.raw_line.set_data(self.steps, self.losses)
        self.smooth_line.set_data(self.steps, self.smoothed_losses)
        self.axis.relim()
        self.axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.figure.savefig(self.output_path, dpi=120)
        plt.pause(0.001)

    def close(self):
        if self.steps:
            self.figure.savefig(self.output_path, dpi=120)
        plt.ioff()
        plt.close(self.figure)
