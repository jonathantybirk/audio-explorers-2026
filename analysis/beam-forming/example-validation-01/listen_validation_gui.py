"""
listen_validation_gui.py  --  Minimal GUI player for example-validation audio

Playback order:
  0deg_front: closest -> geom_das -> mvdr
  90deg_left: closest -> geom_das -> mvdr
  180deg_back: closest -> geom_das -> mvdr
  270deg_right: closest -> geom_das -> mvdr

Uses tkinter when available. If `_tkinter` is missing, falls back to opening
the local HTML player in the browser.
"""

from pathlib import Path
import subprocess
import webbrowser


ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio"
HTML_PLAYER = ROOT / "listen_validation_gui.html"

DIRECTIONS = [
    "0deg_front",
    "90deg_left",
    "180deg_back",
    "270deg_right",
]

METHODS = [
    "closest",
    "geom_das",
    "mvdr",
]


def build_playlist():
    items = []
    for direction in DIRECTIONS:
        for method in METHODS:
            filename = f"{method}_{direction}.wav"
            items.append(
                {
                    "direction": direction,
                    "method": method,
                    "filename": filename,
                    "path": AUDIO_DIR / filename,
                }
            )
    return items


class ValidationPlayer:
    def __init__(self, master):
        self.master = master
        self.master.title("Example Validation Player")
        self.master.geometry("720x420")

        self.playlist = build_playlist()
        self.index = 0
        self.proc = None

        self._build_ui()
        self._populate_list()
        self._select_index(0)
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        main = tk.Frame(self.master, padx=12, pady=12)
        main.pack(fill="both", expand=True)

        title = tk.Label(
            main,
            text="example-validation-01",
            font=("Helvetica", 16, "bold"),
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            main,
            text="Order: closest -> geom_das -> mvdr for each direction",
            font=("Helvetica", 11),
        )
        subtitle.pack(anchor="w", pady=(2, 10))

        self.status_var = tk.StringVar(value="Idle")
        status = tk.Label(main, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", pady=(0, 8))

        info_frame = tk.Frame(main)
        info_frame.pack(fill="x", pady=(0, 8))

        self.direction_var = tk.StringVar(value="")
        self.method_var = tk.StringVar(value="")
        self.file_var = tk.StringVar(value="")

        tk.Label(info_frame, textvariable=self.direction_var, font=("Helvetica", 12, "bold")).pack(anchor="w")
        tk.Label(info_frame, textvariable=self.method_var).pack(anchor="w")
        tk.Label(info_frame, textvariable=self.file_var, fg="#555555").pack(anchor="w")

        list_frame = tk.Frame(main)
        list_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(list_frame, activestyle="dotbox")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)
        self.listbox.bind("<Double-Button-1>", lambda _event: self.play_current())

        controls = tk.Frame(main)
        controls.pack(fill="x", pady=(10, 0))

        for label, cmd in [
            ("Previous", self.previous_item),
            ("Play", self.play_current),
            ("Stop", self.stop_playback),
            ("Replay", self.replay_current),
            ("Next", self.next_item),
        ]:
            tk.Button(controls, text=label, width=10, command=cmd).pack(side="left", padx=(0, 8))

        tk.Label(
            main,
            text="Tips: double-click an item to play it. Use Previous / Next to move through the ordered comparison.",
            fg="#555555",
            anchor="w",
        ).pack(fill="x", pady=(10, 0))

    def _populate_list(self):
        for idx, item in enumerate(self.playlist):
            label = f"{idx + 1:02d}. {item['direction']}  |  {item['method']}"
            self.listbox.insert("end", label)

    def _select_index(self, index):
        if not self.playlist:
            return
        self.index = max(0, min(index, len(self.playlist) - 1))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.index)
        self.listbox.activate(self.index)
        self.listbox.see(self.index)
        self._refresh_labels()

    def _refresh_labels(self):
        item = self.playlist[self.index]
        self.direction_var.set(f"Direction: {item['direction']}")
        self.method_var.set(f"Method: {item['method']}")
        self.file_var.set(f"File: audio/{item['filename']}")

    def on_list_select(self, _event=None):
        selection = self.listbox.curselection()
        if not selection:
            return
        self.index = int(selection[0])
        self._refresh_labels()

    def _play_path(self, path):
        self.stop_playback()
        if not path.exists():
            self.messagebox.showerror("Missing file", f"Could not find:\n{path}")
            return
        self.proc = subprocess.Popen(["afplay", str(path)])
        self.status_var.set(f"Playing: {path.name}")
        self.master.after(300, self._poll_proc)

    def _poll_proc(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.master.after(300, self._poll_proc)
            return
        finished_name = self.playlist[self.index]["filename"]
        self.proc = None
        self.status_var.set(f"Finished: {finished_name}")

    def play_current(self):
        self._play_path(self.playlist[self.index]["path"])

    def replay_current(self):
        self.play_current()

    def stop_playback(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        self.proc = None
        self.status_var.set("Stopped")

    def previous_item(self):
        self._select_index(self.index - 1)

    def next_item(self):
        self._select_index(self.index + 1)

    def on_close(self):
        self.stop_playback()
        self.master.destroy()


def launch_tkinter():
    import tkinter as tk
    from tkinter import messagebox

    ValidationPlayer.tk = tk
    ValidationPlayer.messagebox = messagebox
    globals()["tk"] = tk

    root = tk.Tk()
    ValidationPlayer(root)
    root.mainloop()


def launch_html_fallback():
    webbrowser.open(HTML_PLAYER.resolve().as_uri())
    print(f"Opened browser player: {HTML_PLAYER}")


def main():
    try:
        launch_tkinter()
    except ModuleNotFoundError as exc:
        if exc.name != "_tkinter":
            raise
        launch_html_fallback()


if __name__ == "__main__":
    main()
