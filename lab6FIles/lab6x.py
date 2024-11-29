import tkinter as tk
from tkinter import ttk
import customtkinter
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Konfiguracja loggera
logging.basicConfig(
    filename='logger.txt',
    level=logging.INFO,
    format='%(message)s',
    filemode='w'
)
logger = logging.getLogger("")
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

class MLApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # App configuration
        self.title("ML Logit")
        self.geometry("900x600")
        customtkinter.set_appearance_mode("Light")  # Default appearance mode
        customtkinter.set_default_color_theme("blue")

        # Sidebar frame
        self.sidebar_frame = customtkinter.CTkFrame(self, width=200)
        self.sidebar_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Buttons on sidebar
        self.sidebar_button_1 = customtkinter.CTkButton(
            self.sidebar_frame, text="Train", command=self.model_train
        )
        self.sidebar_button_1.grid(row=0, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(
            self.sidebar_frame, text="Predict", command=self.predict_value
        )
        self.sidebar_button_2.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_3 = customtkinter.CTkButton(
            self.sidebar_frame, text="Plot", command=self.plot_graph
        )
        self.sidebar_button_3.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_4 = customtkinter.CTkButton(
            self.sidebar_frame, text="Load CSV", command=self.load_csv
        )
        self.sidebar_button_4.grid(row=3, column=0, padx=20, pady=10)

        # Appearance settings
        self.appearance_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="Appearance Mode:"
        )
        self.appearance_label.grid(row=4, column=0, padx=20, pady=10)

        self.appearance_mode = customtkinter.CTkOptionMenu(
            self.sidebar_frame,
            values=["Light", "Dark"],
            command=self.change_appearance_mode,
        )
        self.appearance_mode.grid(row=5, column=0, padx=20, pady=10)

        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:")
        self.scaling_label.grid(row=6, column=0, padx=20, pady=10)

        self.ui_scaling = customtkinter.CTkOptionMenu(
            self.sidebar_frame,
            values=["80%", "100%", "120%"],
            command=self.change_scaling,
        )
        self.ui_scaling.grid(row=7, column=0, padx=20, pady=10)

        # Canvas frame
        self.canvas_frame = customtkinter.CTkFrame(self)
        self.canvas_frame.grid(row=0, column=1, padx=(20, 10), pady=10, sticky="nsew")

        # CSV Table
        self.csvtable = ttk.Treeview(self, show="headings")
        self.csvtable.grid(row=0, column=2, padx=(10, 20), pady=10, sticky="nsew")

        # Initialize data and model
        self.data = pd.DataFrame(columns=["x", "y"])
        self.model = LinearRegression()

    def model_train(self):
        if not self.data.empty:
            x = self.data["x"].values.reshape(-1, 1)
            y = self.data["y"].values
            self.model.fit(x, y)
            customtkinter.CTkMessageBox.show_info("Training complete!")

    def predict_value(self):
        x_new = float(self.sidebar_frame.winfo_children()[-1].get())
        y_pred = self.model.predict([[x_new]])
        customtkinter.CTkMessageBox.show_info(f"Predicted value: {y_pred[0]}")

    def plot_graph(self):
        if not self.data.empty:
            fig = Figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            ax.scatter(self.data["x"], self.data["y"], color="blue", label="Data Points")
            ax.plot(
                self.data["x"],
                self.model.predict(self.data["x"].values.reshape(-1, 1)),
                color="red",
                label="Regression Line",
            )
            ax.set_title("Linear Regression")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(padx=(20, 20), pady=(20, 20))

    def load_csv(self):
        self.data = pd.read_csv("data.csv")
        for col in self.data.columns:
            self.csvtable.heading(col, text=col)
        for _, row in self.data.iterrows():
            self.csvtable.insert("", "end", values=row.tolist())

    def change_appearance_mode(self, mode):
        customtkinter.set_appearance_mode(mode)

    def change_scaling(self, scaling):
        new_scaling = int(scaling.strip("%")) / 100
        self.tk.call("tk", "scaling", new_scaling)


if __name__ == "__main__":
    app = MLApp()
    app.mainloop()
