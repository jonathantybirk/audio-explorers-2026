"""
listen_validation.py  --  Step through example-validation audio in repo order

Plays:
  direction 1: closest -> geom_das -> mvdr
  direction 2: closest -> geom_das -> mvdr
  ...

Uses macOS `afplay` for playback.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio"

directions = [
    "0deg_front",
    "90deg_left",
    "180deg_back",
    "270deg_right",
]

methods = [
    "closest",
    "geom_das",
    "mvdr",
]


def main():
    print("Press Enter to play each file, or type q then Enter to stop.\n")

    for direction in directions:
        print(f"=== {direction} ===")
        for method in methods:
            filename = f"{method}_{direction}.wav"
            path = AUDIO_DIR / filename
            print(f"\nNext: {filename}")
            user = input("Enter = play, q = quit: ").strip().lower()
            if user == "q":
                print("Stopped.")
                return
            subprocess.run(["afplay", str(path)], check=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
