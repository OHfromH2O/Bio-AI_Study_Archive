"""
Handwritten Digit Recognition Application
CNN trained on MNIST dataset + Tkinter drawing canvas GUI.

Features:
  - Draw digits 0-9 with the mouse
  - Auto-predicts on every mouse release
  - Shows confidence bars for all 10 digits
  - Clear button to reset the canvas
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

CANVAS_SIZE = 280    # display canvas size (pixels)
BRUSH_SIZE  = 10     # half-width of the drawing brush (pixels)


# ─────────────────────────────────────────────────────────────────────────────
#  Model — build, train, predict
# ─────────────────────────────────────────────────────────────────────────────

class DigitRecognizer:
    """Loads MNIST, trains a CNN, and exposes predict_digit()."""

    def __init__(self):
        self.model = self._build_and_train()

    # ------------------------------------------------------------------
    def _build_and_train(self) -> keras.Model:
        # Load MNIST and normalise to [0, 1]
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
        x_test  = x_test.astype("float32")[..., np.newaxis]  / 255.0

        # CNN: two conv blocks → flatten → dropout → dense output
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation="relu",
                                input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation="softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("Training CNN on MNIST (5 epochs) ...")
        model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=128,
            validation_data=(x_test, y_test),
            verbose=1,
        )

        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {acc:.4f}\n")
        return model

    # ------------------------------------------------------------------
    def predict_digit(self, pil_image: Image.Image):
        """
        Preprocess a PIL image and return (predicted_digit, prob_array).

        The canvas has a white background with black strokes, so colours
        are inverted before being fed to the model (MNIST uses white
        digits on a black background).
        """
        img = pil_image.convert("L").resize((28, 28), Image.LANCZOS)
        arr = np.array(img, dtype="float32")
        arr = (255.0 - arr) / 255.0                    # invert + normalise
        arr = arr.reshape(1, 28, 28, 1)

        probs = self.model.predict(arr, verbose=0)[0]
        return int(np.argmax(probs)), probs


# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────

class DrawingApp:
    """Tkinter window with a drawing canvas and real-time prediction panel."""

    def __init__(self, root: tk.Tk, recognizer: DigitRecognizer):
        self.root       = root
        self.recognizer = recognizer
        self.root.title("Handwritten Digit Recognition")
        self.root.resizable(False, False)
        self.root.configure(bg="#f5f5f5")

        # Off-screen PIL buffer (mirrors the Tkinter canvas)
        self._reset_buffer()

        self._build_ui()
        self._bind_events()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _reset_buffer(self):
        self.pil_image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 12, "pady": 12}

        # ── LEFT column: drawing canvas ────────────────────────────────
        left = tk.Frame(self.root, bg="#f5f5f5")
        left.grid(row=0, column=0, **pad, sticky="n")

        tk.Label(left, text="Draw a digit  (0 – 9)",
                 font=("Helvetica", 13, "bold"),
                 bg="#f5f5f5", fg="#333").pack(pady=(0, 6))

        self.canvas = tk.Canvas(
            left,
            width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="white", cursor="crosshair",
            highlightthickness=2, highlightbackground="#aaa",
        )
        self.canvas.pack()

        # Buttons
        btn_row = tk.Frame(left, bg="#f5f5f5")
        btn_row.pack(pady=10)

        tk.Button(
            btn_row, text="Predict",
            command=self._predict,
            width=10, bg="#4CAF50", fg="white",
            font=("Helvetica", 11, "bold"), relief="flat",
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_row, text="Clear",
            command=self._clear,
            width=10, bg="#f44336", fg="white",
            font=("Helvetica", 11, "bold"), relief="flat",
        ).pack(side=tk.LEFT, padx=6)

        # ── RIGHT column: prediction results ───────────────────────────
        right = tk.Frame(self.root, bg="#f5f5f5")
        right.grid(row=0, column=1, **pad, sticky="n")

        tk.Label(right, text="Prediction",
                 font=("Helvetica", 13, "bold"),
                 bg="#f5f5f5", fg="#333").pack(pady=(0, 6))

        # Large digit display
        self.result_var = tk.StringVar(value="?")
        tk.Label(
            right, textvariable=self.result_var,
            font=("Helvetica", 80, "bold"),
            fg="#222", bg="white", width=3,
            relief="groove",
        ).pack()

        # Separator
        tk.Frame(right, height=2, bg="#ccc").pack(fill=tk.X, pady=10)

        tk.Label(right, text="Confidence per digit:",
                 font=("Helvetica", 10), bg="#f5f5f5", fg="#555").pack(anchor="w")

        # One progress bar per digit (0–9)
        self.bars    = {}
        self.pct_var = {}
        for d in range(10):
            row = tk.Frame(right, bg="#f5f5f5")
            row.pack(fill=tk.X, pady=2)

            tk.Label(row, text=str(d), width=2,
                     font=("Helvetica", 10, "bold"),
                     bg="#f5f5f5", fg="#333").pack(side=tk.LEFT)

            bar = ttk.Progressbar(row, length=200, maximum=100,
                                  style="TProgressbar")
            bar.pack(side=tk.LEFT, padx=4)

            pv = tk.StringVar(value="0.0%")
            tk.Label(row, textvariable=pv, width=6,
                     font=("Helvetica", 9),
                     bg="#f5f5f5", fg="#555").pack(side=tk.LEFT)

            self.bars[d]    = bar
            self.pct_var[d] = pv

    # ── event handling ────────────────────────────────────────────────────────

    def _bind_events(self):
        self.canvas.bind("<Button-1>",        self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self._prev = None

    def _on_press(self, event):
        self._prev = (event.x, event.y)

    def _on_drag(self, event):
        x, y = event.x, event.y
        if self._prev:
            px, py = self._prev
            # Draw on Tkinter canvas
            self.canvas.create_line(
                px, py, x, y,
                fill="black", width=BRUSH_SIZE * 2,
                capstyle=tk.ROUND, smooth=True,
            )
            # Mirror on PIL buffer
            r = BRUSH_SIZE
            self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill="black")
            self.pil_draw.line([px, py, x, y],
                               fill="black", width=BRUSH_SIZE * 2)
        self._prev = (x, y)

    def _on_release(self, _event):
        self._prev = None
        self._predict()          # auto-predict on every stroke

    # ── actions ───────────────────────────────────────────────────────────────

    def _predict(self):
        digit, probs = self.recognizer.predict_digit(self.pil_image)
        self.result_var.set(str(digit))

        for d, prob in enumerate(probs):
            pct = prob * 100
            self.bars[d]["value"] = pct
            self.pct_var[d].set(f"{pct:.1f}%")
            # Highlight the winning bar in green, others in default blue
            self.bars[d].config(
                style="green.Horizontal.TProgressbar" if d == digit
                      else "TProgressbar"
            )

    def _clear(self):
        self.canvas.delete("all")
        self._reset_buffer()
        self.result_var.set("?")
        for d in range(10):
            self.bars[d]["value"] = 0
            self.pct_var[d].set("0.0%")
            self.bars[d].config(style="TProgressbar")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Starting Handwritten Digit Recognition App ...")
    print("Training a CNN on MNIST -- please wait.\n")

    # Build root window first so ttk styles can be registered
    root = tk.Tk()

    style = ttk.Style(root)
    style.theme_use("default")
    style.configure(
        "green.Horizontal.TProgressbar",
        troughcolor="#e0e0e0",
        background="#4CAF50",
    )
    style.configure(
        "TProgressbar",
        troughcolor="#e0e0e0",
        background="#2196F3",
    )

    recognizer = DigitRecognizer()     # downloads MNIST & trains the model
    DrawingApp(root, recognizer)
    root.mainloop()


if __name__ == "__main__":
    main()
