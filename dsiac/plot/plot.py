import numpy as np


class Index:
    def __init__(self, fptr, frames, ax, fig):
        self.fptr = fptr
        self.frames = frames
        self.ax = ax
        self.selected = 0
        self.fig = fig

    def _render(self):
        im = self.fptr[self.selected].byteswap()
        vmin, vmax = np.percentile(im, [0.5, 99.5])
        self.ax.cla()
        self.ax.imshow(im, vmin=vmin, vmax=vmax, cmap="gray")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def next(self, _event):
        self.selected = min(self.frames - 1, self.selected + 1)

    def prev(self, _event):
        self.selected = max(self.selected - 1, 0)
