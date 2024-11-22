import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging

# Konfiguracja loggera
logging.basicConfig(
    filename='logger.txt',
    level=logging.INFO,
    format='%(message)s',
    filemode='w'
)
logger = logging.getLogger("")


# LAB6 funkcje
def predict_value():
    X_unknown = np.array([2.78])
    return imported_model.coef_[0][0] * X_unknown[0] + imported_model.intercept_[0]


def update_and_fit_model(csv_path, x, y):
    df = pd.read_csv(csv_path)
    # Dodaj nowy wiersz do danych
    df.loc[len(df.index)] = [x, y]
    df.to_csv(csv_path, index=False)
    # Odczytaj dane ponownie
    df = pd.read_csv(csv_path)
    # Przygotuj dane
    x = df['x'].values.reshape(-1, 1)
    y = df['y'].values.reshape(-1, 1)
    # Utwórz i trenuj model
    our_model = LinearRegression()
    our_model.fit(x, y)
    # Zapisz model do pliku
    pickle.dump(our_model, open('our_model.pkl', 'wb'))


# INTERFEJS
class MLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Logit - Tkinter")
        self.geometry("900x600")

        # Frame na przyciski (sidebar)
        self.sidebar_frame = tk.Frame(self, width=200, bg="lightgrey")
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Przyciski w sidebar
        self.update_button = tk.Button(self.sidebar_frame, text="Update & Train", command=self.update_and_train)
        self.update_button.pack(pady=10, padx=20)

        self.plot_button = tk.Button(self.sidebar_frame, text="Plot", command=self.plot_data)
        self.plot_button.pack(pady=10, padx=20)

        # Pola tekstowe do wprowadzania X i Y
        self.x_label = tk.Label(self.sidebar_frame, text="X Value:")
        self.x_label.pack(pady=5, padx=20)
        self.x_entry = tk.Entry(self.sidebar_frame)
        self.x_entry.pack(pady=5, padx=20)

        self.y_label = tk.Label(self.sidebar_frame, text="Y Value:")
        self.y_label.pack(pady=5, padx=20)
        self.y_entry = tk.Entry(self.sidebar_frame)
        self.y_entry.pack(pady=5, padx=20)

        # Okno na wykres
        self.canvas_frame = tk.Frame(self, bg="white")
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tabela CSV
        self.table_frame = tk.Frame(self, bg="white")
        self.table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.csv_table = ttk.Treeview(self.table_frame, columns=("x", "y"), show="headings")
        self.csv_table.heading("x", text="X")
        self.csv_table.heading("y", text="Y")
        self.csv_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Inicjalizacja danych
        self.csv_path = '10_points.csv'
        self.model = None

    def load_csv(self):
        try:
            df = pd.read_csv(self.csv_path)
            for row in self.csv_table.get_children():
                self.csv_table.delete(row)
            for _, row in df.iterrows():
                self.csv_table.insert("", "end", values=(row["x"], row["y"]))
            messagebox.showinfo("Info", "CSV Loaded Successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def update_and_train(self):
        try:
            # Pobierz wartości z pól tekstowych
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            update_and_fit_model(self.csv_path, x, y)
            messagebox.showinfo("Info", "Data updated and model retrained successfully!")
            self.load_csv()  # Update the table in the app
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for X and Y.")

    def plot_data(self):
        df = pd.read_csv(self.csv_path)
        if self.model is None:
            self.model = pickle.load(open("our_model.pkl", "rb"))
        x = df['x'].values.reshape(-1, 1)
        y = df['y'].values

        # Tworzenie wykresu
        fig = Figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(x, y, color="blue", label="Data Points")
        ax.plot(x, self.model.predict(x), color="red", label="Regression Line")
        ax.set_title("Linear Regression")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

        # Wyświetlanie wykresu
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    imported_model = pickle.load(open("our_model.pkl", "rb"))
    app = MLApp()
    app.mainloop()
