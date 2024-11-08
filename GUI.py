import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GuiAgain(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Task 1")
        self.geometry("1900x900")
        self.container = Container(self)
        self.plot = Plot(self)
        self.mainloop()


class Container(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.place(relx=0, rely=0, relwidth=0.45, relheight=1)
        self.input_parameter = InputParameter(self)
        self.output_parameter = OutputParameter(self)
        self.input_parameter.create_widgets()
        self.output_parameter.create_widgets()


class InputParameter(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.mse_threshold = None
        self.learning_rate = None
        self.classes_combobox = None
        self.feature_combobox = None
        self.num_of_epoch = None
        self.selected_model = tk.IntVar()
        self.bias_var = tk.BooleanVar()
        self.place(relx=0, rely=0, relwidth=1, relheight=0.5)

    def create_widgets(self):
        # feature
        (tk.Label(self, text="Select Features:", font=('JetBrains Mono', 15)).grid(row=1, column=0, sticky='e', padx=10,
                                                                                   pady=5))
        self.feature_combobox = ttk.Combobox(self, values=["Gender & Body Mass",
                                                           "Gender & Beak Length",
                                                           "Gender & Beak Depth",
                                                           "Gender & Fin Length",
                                                           "Body Mass & Beak Length",
                                                           "Body Mass & Beak Depth",
                                                           "Body Mass & Fin Length",
                                                           "Beak Length & Beak Depth",
                                                           "Beak Length & Fin Length",
                                                           "Beak Depth & Fin Length"]
                                             , font=('JetBrains Mono', 15), width=25)
        self.feature_combobox.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        # class
        tk.Label(self, text="Select Classes:", font=('JetBrains Mono', 15)).grid(row=2, column=0, sticky='e',
                                                                                 padx=10, pady=5)
        self.classes_combobox = ttk.Combobox(self, values=["C1 & C2", "C2 & C3", "C1 & C3"],
                                             font=('JetBrains Mono', 15), width=25)
        self.classes_combobox.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        # model
        tk.Label(self, text="Select Model:", font=('JetBrains Mono', 15)).grid(row=3, column=0, sticky='e',
                                                                               padx=5, pady=5)
        tk.Radiobutton(self, text="Perceptron", variable=self.selected_model, value=1,
                       font=('JetBrains Mono', 15)).grid(row=3, column=1, sticky='w')
        tk.Radiobutton(self, text="Adaline", variable=self.selected_model, value=2, font=('JetBrains Mono', 15)).grid(
            row=3, column=2, sticky='w')
        # epoch
        tk.Label(self, text="Number of Epochs (m):", font=('JetBrains Mono', 15)).grid(row=4, column=0,
                                                                                       sticky='e', padx=5,
                                                                                       pady=5)
        self.num_of_epoch = tk.Entry(self, width=20, font=('JetBrains Mono', 15))
        self.num_of_epoch.grid(row=4, column=1, padx=5, pady=5)
        # learning rate
        tk.Label(self, text="Learning Rate (eta):", font=('JetBrains Mono', 15)).grid(row=5, column=0,
                                                                                      sticky='e', padx=5, pady=5)
        self.learning_rate = tk.Entry(self, width=20, font=('JetBrains Mono', 15))
        self.learning_rate.grid(row=5, column=1, padx=5, pady=5)
        # mse
        tk.Label(self, text="MSE Threshold (mse_threshold):", font=('JetBrains Mono', 15)).grid(row=6, column=0,
                                                                                                sticky='e',
                                                                                                padx=5, pady=5)
        self.mse_threshold = tk.Entry(self, width=20, font=('JetBrains Mono', 15))
        self.mse_threshold.grid(row=6, column=1, padx=5, pady=5)
        # Add Bias
        tk.Label(self, text="Add Bias:", font=('JetBrains Mono', 15)).grid(row=7, column=0, sticky='e', padx=5,
                                                                           pady=5)
        tk.Checkbutton(self, variable=self.bias_var, font=('JetBrains Mono', 15)).grid(row=7, column=1, pady=10)
        # run & exit buttons
        tk.Button(self, text="Run", command=self.run, padx=25, pady=10, background='blue',
                  font=('JetBrains Mono', 10)).grid(row=8, column=0, padx=10, pady=5)

        tk.Button(self, text="Exit", command=self.quit, padx=25, pady=10, background='red',
                  font=('JetBrains Mono', 10)).grid(row=8, column=1, padx=10, pady=5)

    def run(self):
        print("I'm here in run function")
        print("Selected feature:", self.feature_combobox.get())
        print("Selected classes:", self.classes_combobox.get())
        print("Selected model:", self.selected_model.get())
        print("Number of epochs:", self.num_of_epoch.get())
        print("Learning rate:", self.learning_rate.get())
        print("MSE threshold:", self.mse_threshold.get())
        print("Add Bias:", self.bias_var.get())

        if (not self.feature_combobox.get() or not self.classes_combobox.get() or self.selected_model.get() == 0
                or not self.num_of_epoch.get() or not self.learning_rate.get() or not self.mse_threshold.get()):
            messagebox.showerror("Input error", "You must enter all required values")
            return

class OutputParameter(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.matrix_entries = None
        self.overall_entry = None
        self.place(relx=0, rely=0.5, relwidth=1, relheight=0.5)

    def create_widgets(self):
        tk.Label(self, text="Output", font=('JetBrains Mono', 20, 'bold')).grid(row=0, column=0, padx=10,
                                                                                pady=5)
        tk.Label(self, text="Overall Accuracy:", font=('JetBrains Mono', 15, 'bold')).grid(row=5, column=0,
                                                                                           padx=10, pady=5)
        self.overall_entry = tk.Entry(self, width=20, font=('JetBrains Mono', 15))
        self.overall_entry.grid(row=5, column=1, padx=5, pady=5)

        tk.Label(self, text="Predicted", font=('JetBrains Mono', 15, 'bold')).grid(row=1, column=1,
                                                                                   columnspan=2, padx=10,
                                                                                   pady=5)
        tk.Label(self, text="Actual", font=('JetBrains Mono', 15, 'bold')).grid(row=1, column=0, rowspan=2,
                                                                                padx=10, pady=5)

        tk.Label(self, text="Positive", font=('JetBrains Mono', 12)).grid(row=2, column=1, padx=5, pady=5)
        tk.Label(self, text="Negative", font=('JetBrains Mono', 12)).grid(row=2, column=2, padx=5, pady=5)

        tk.Label(self, text="Positive", font=('JetBrains Mono', 12)).grid(row=3, column=0, padx=5, pady=5)
        tk.Label(self, text="Negative", font=('JetBrains Mono', 12)).grid(row=4, column=0, padx=5, pady=5)

        self.matrix_entries = self.create_matrix_entries(self)

    def create_matrix_entries(self, parent):
        labels = [['TP', 'FP'], ['FN', 'TN']]
        matrix_entries = []
        for i in range(2):
            row_entries = []
            for j in range(2):
                cell_frame = tk.Frame(parent)
                cell_frame.grid(row=i + 3, column=j + 1, padx=5, pady=5)
                tk.Label(cell_frame, text=labels[i][j], font=('JetBrains Mono', 10)).pack(anchor='n')
                entry = tk.Entry(cell_frame, width=15, font=('JetBrains Mono', 15), justify='center', state='readonly')
                entry.pack(anchor='s')
                row_entries.append(entry)
            matrix_entries.append(row_entries)
        return matrix_entries

    def set_matrix_values(self, tp=0, tn=0, fp=0, fn=0):
        values = [[tp, fp], [fn, tn]]
        for i in range(2):
            for j in range(2):
                var = tk.StringVar()
                self.matrix_entries[i][j].config(textvariable=var)
                var.set(str(values[i][j]))


class Plot(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.place(relx=0.45, rely=0, relwidth=0.55, relheight=1)
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(expand=True, fill='both')

    def visualize_classes(self, x_train, y_train, weights, bias, features, title="Birds Dataset"):
        self.ax.clear()
        features_name = ["Gender", "Body Mass", "Beak Length", "Beak Depth", "Fin Length"]
        feature1, feature2 = features
        class0 = x_train[y_train == 0]
        class1 = x_train[y_train == 1]

        self.ax.scatter(class0[:, feature1], class0[:, feature2],
                        color='red', label='Class 0', alpha=0.7, s=100)
        self.ax.scatter(class1[:, feature1], class1[:, feature2],
                        color='black', label='Class 1', alpha=0.7, s=100)

        x1 = np.linspace(x_train[:, feature1].min(), x_train[:, feature1].max(), 100)
        x2 = -(weights[0] * x1 + bias) / weights[1]
        self.ax.plot(x1, x2, 'blue', label='Decision Boundary')
        self.ax.set_xlabel(features_name[feature1])
        self.ax.set_ylabel(features_name[feature2])
        self.ax.set_title(title)
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.legend()
        self.canvas.draw()
