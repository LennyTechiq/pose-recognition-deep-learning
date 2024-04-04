import tkinter as tk
from tkinter import simpledialog, messagebox
import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model

import mediapipe as mp

class DanceMoveRecognizerApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Set the dimensions of the window
        window_width = 800
        window_height = 600

        # Get the screen width and height
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # Calculate the x and y coordinates to center the window
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)

        # Set the window to be centered
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Add a background color to the window
        self.window.configure(bg='#f0f0f0')

        # Create a label for the title
        self.lbl_title = tk.Label(window, text="Dance Move Recognition System", font=("Helvetica", 20), bg='#f0f0f0', pady=20)
        self.lbl_title.pack()


        self.video_source = 0  # You might need to change this depending on your camera setup
        self.vid = cv2.VideoCapture(self.video_source)

        # Set the dimensions for the canvas
        canvas_width = 500
        canvas_height = 400

        self.canvas = tk.Canvas(window, width=canvas_width, height=canvas_height)
        self.canvas.pack()

        # Create buttons with some styling
        self.btn_frame = tk.Frame(window, bg='#f0f0f0')
        self.btn_frame.pack()

        self.btn_start_collect = tk.Button(self.btn_frame, text="Start Data Collection", width=20, command=self.start_data_collection, bg='#4CAF50', fg='white', font=("Helvetica", 14), pady=10)
        self.btn_start_collect.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_train_model = tk.Button(self.btn_frame, text="Train Model", width=20, command=self.train_model, state=tk.DISABLED, bg='#2196F3', fg='white', font=("Helvetica", 14), pady=10)
        self.btn_train_model.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_recognize_moves = tk.Button(self.btn_frame, text="Recognize Dance Moves", width=20, command=self.recognize_moves, bg='#FF5722', fg='white', font=("Helvetica", 14), pady=10)
        self.btn_recognize_moves.pack(side=tk.LEFT, padx=10, pady=10)




        self.delay = 10  # milliseconds
        self.update()

        self.window.mainloop()

    def start_data_collection(self):
        name = simpledialog.askstring("Input", "Enter the name of the Dance Move:")

        holistic = mp.solutions.pose
        holis = holistic.Pose()
        drawing = mp.solutions.drawing_utils

        X = []
        data_size = 0

        # Get the dimensions of the screen
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # Calculate the center coordinates
        center_x = screen_width // 2
        center_y = screen_height // 2

        # Calculate the dimensions for the camera window
        window_width = 800
        window_height = 600

        # Calculate the position for the camera window to be centered
        window_x = center_x - (window_width // 2)
        window_y = center_y - (window_height // 2)

        while True:
            lst = []

            _, frm = self.vid.read()

            frm = cv2.flip(frm, 1)

            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            if res.pose_landmarks and self.inFrame(res.pose_landmarks.landmark):
                for i in res.pose_landmarks.landmark:
                    lst.append(i.x - res.pose_landmarks.landmark[0].x)
                    lst.append(i.y - res.pose_landmarks.landmark[0].y)

                X.append(lst)
                data_size += 1
            else:
                cv2.putText(frm, "Make Sure Full body visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)

            cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

             # Resize the frame to desired dimensions
            frm = cv2.resize(frm, (window_width, window_height))

            # Create a black background to display the frame at the center
            background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            # Position the frame at the center of the background
            background[window_y:window_y + window_height, window_x:window_x + window_width] = frm


            cv2.imshow("window", frm)

            if cv2.waitKey(1) == 27 or data_size > 80:
                break

        np.save(f"{name}.npy", np.array(X))
        print("Data collection completed.")
        self.btn_train_model.config(state=tk.NORMAL)

    def inFrame(self, lst):
        if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
            return True
        return False

    def train_model(self):
        X = []
        y = []

        for filename in os.listdir():
            if filename.endswith(".npy") and filename != "labels.npy":
                data = np.load(filename)
                X.extend(data)
                y.extend([filename[:-4]] * len(data))

        X = np.array(X)
        y = np.array(y)

        label_dict = {label: i for i, label in enumerate(np.unique(y))}
        y = np.array([label_dict[label] for label in y])
        y = to_categorical(y)

        X, y = self.shuffle_data(X, y)

        input_layer = Input(shape=(X.shape[1],))
        dense1 = Dense(128, activation="tanh")(input_layer)
        dense2 = Dense(64, activation="tanh")(dense1)
        output_layer = Dense(y.shape[1], activation="softmax")(dense2)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])
        model.fit(X, y, epochs=80)

        model.save("model.h5")
        np.save("labels.npy", np.array(label_dict))

        messagebox.showinfo("Training Completed", "Model training completed successfully!")

    def recognize_moves(self):
        model = load_model("model.h5")
        label = np.load("labels.npy")

        holistic = mp.solutions.pose
        holis = holistic.Pose()
        drawing = mp.solutions.drawing_utils

        while True:
            lst = []

            ret, frm = self.vid.read()

            window = np.zeros((1200, 1600, 3), dtype="uint8")

            frm = cv2.flip(frm, 1)

            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            frm = cv2.blur(frm, (4, 4))
            if res.pose_landmarks and self.inFrame(res.pose_landmarks.landmark):
                for i in res.pose_landmarks.landmark:
                    lst.append(i.x - res.pose_landmarks.landmark[0].x)
                    lst.append(i.y - res.pose_landmarks.landmark[0].y)

                lst = np.array(lst).reshape(1, -1)

                p = model.predict(lst)
                pred = label[np.argmax(p)]

                if p[0][np.argmax(p)] > 0.75:
                    cv2.putText(window, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)
                else:
                    cv2.putText(window, "Dance move is either wrong not trained", (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)

            else:
                cv2.putText(frm, "Make Sure Full body visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                   connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                                   landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3,
                                                                              thickness=3))

            window[200:800, 400:1200, :] = cv2.resize(frm, (800, 600))

            cv2.imshow("window", window)

            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                self.vid.release()
                break

    def shuffle_data(self, X, y):
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = cv2.resize(self.photo, (int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            self.photo = tk.PhotoImage(data=cv2.imencode('.png', self.photo)[1].tobytes())
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    app = DanceMoveRecognizerApp(root, "Dance Move Recognition System")
