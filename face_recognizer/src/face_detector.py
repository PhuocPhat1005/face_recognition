import tkinter as tk
from tkinter import messagebox
from insightface.app import FaceAnalysis
import cv2 as cv
import numpy as np
from numpy.linalg import norm
import os
import csv
from datetime import datetime

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Recognition App")

        # Initialize FaceAnalysis globally
        self.face_analysis = FaceAnalysis(
            name="buffalo_l",
            allow_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 480))

        # Load registered faces from the "register/embeddings" folder
        self.registered_faces = self.load_embeddings_from_folder("register/embeddings")

        self.label = tk.Label(master, text="Choose an action:")
        self.label.pack()

        self.register_button = tk.Button(master, text="Register", command=self.face_register)
        self.register_button.pack()

        self.verify_button = tk.Button(master, text="Verify", command=self.face_verify)
        self.verify_button.pack()

        self.attendance_button = tk.Button(master, text="Attendance", command=self.face_attendance)
        self.attendance_button.pack()

    def load_embeddings_from_folder(self, folder_path):
        registered_faces = []

        for user_name in os.listdir(folder_path):
            user_folder = os.path.join(folder_path, user_name)

            if os.path.isdir(user_folder):
                for file_name in os.listdir(user_folder):
                    if file_name.endswith(".npy"):
                        embedding_path = os.path.join(user_folder, file_name)
                        embedding = np.load(embedding_path)
                        registered_faces.append({"user_name": user_name, "embedding": embedding})

        return registered_faces

    def calculate_similarity(self, embeddings1, embeddings2):
        similarity = np.dot(embeddings1, embeddings2) / (norm(embeddings1) * norm(embeddings2))
        return similarity

    def is_user_registered(self, user_name):
        for reg_face in self.registered_faces:
            if user_name.lower() == reg_face["user_name"].lower():
                return True
        return False

    def save_embedding_matrix(self, user_name, embedding_matrix):
        result_folder = "register/results"
        os.makedirs(result_folder, exist_ok=True)
        result_file_path = os.path.join(result_folder, f"{user_name}_embedding_matrix.npy")
        np.save(result_file_path, embedding_matrix)

    def save_registration_to_csv(self, user_name, email):
        registration_data = [user_name, email, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S")]
        csv_file_path = "registration_data.csv"

        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if os.path.getsize(csv_file_path) == 0:
                writer.writerow(["Name", "Email", "Date", "Time"])
            writer.writerow(registration_data)

    def face_register(self):
        user_name = input("Enter your name (or 'q' to quit): ")
        if user_name.lower() == 'q':
            return
        if self.is_user_registered(user_name):
            messagebox.showerror("Error", f"This person ({user_name}) is already registered.")
            return

        email = input("Enter your email: ")
        self.face_register_core(user_name, email)

    def face_register_core(self, user_name, email):
        user_folder = os.path.join("register/embeddings", user_name)
        os.makedirs(user_folder, exist_ok=True)

        user_folder_pics = os.path.join("register/pics", user_name)
        os.makedirs(user_folder_pics, exist_ok=True)

        cap = cv.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        embedding_matrix = []

        for index in range(3):
            while True:
                ret, frame = cap.read()
                frame_copy = frame.copy()

                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                faces = self.face_analysis.get(frame_copy)

                for idx, face in enumerate(faces):
                    bbox = face.bbox.astype(int)

                    cv.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                    if face.kps is not None:
                        landmarks = face.kps.astype(int)

                        for point in landmarks:
                            cv.circle(frame_copy, (point[0], point[1]), 2, (0, 0, 255), 2)

                if index == 0:
                    cv.putText(frame_copy, "Frontal_Face", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                elif index == 1:
                    cv.putText(frame_copy, "Left_Face", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                elif index == 2:
                    cv.putText(frame_copy, "Right_Face", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

                cv.imshow('Face-Register', frame_copy)

                key = cv.waitKey(1)
                if key == ord('c'):
                    user_index = index + 1

                    # Check for similarity with registered faces
                    if self.check_for_similarity(user_name, faces[0].embedding):
                        messagebox.showerror("Error", f"This person ({user_name}) is already registered.")
                        return

                    np.save(os.path.join(user_folder, f"{user_name}_{user_index}.npy"), faces[0].embedding)
                    cv.imwrite(os.path.join(user_folder_pics, f"{user_name}_{user_index}.jpg"), frame)
                    print(f"Face and landmarks of {user_name} captured and saved as {user_name}_{user_index}.jpg")

                    self.save_registration_to_csv(user_name, email)

                    embedding_matrix.append(faces[0].embedding)

                    break

                elif key == ord('q'):
                    break

        cap.release()
        cv.destroyAllWindows()

        if embedding_matrix:
            # Check for similarity again before saving the embedding matrix
            if self.check_for_similarity_before_save(user_name, embedding_matrix):
                messagebox.showerror("Error", f"This person ({user_name}) is already registered.")
                return

            embedding_matrix = np.stack(embedding_matrix)
            self.save_embedding_matrix(user_name, embedding_matrix)

    def face_verify(self):
        cap = cv.VideoCapture(0)

        registered_embeddings = np.array([face["embedding"] for face in self.registered_faces])
        registered_user_names = [face["user_name"] for face in self.registered_faces]

        while True:
            ret, frame = cap.read()
            frame_copy = frame.copy()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            faces = self.face_analysis.get(frame_copy)

            for idx, face in enumerate(faces):
                bbox = face.bbox.astype(int)

                cv.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                if face.kps is not None:
                    landmarks = face.kps.astype(int)

                    for point in landmarks:
                        cv.circle(frame_copy, (point[0], point[1]), 2, (0, 0, 255), 2)

                if len(self.registered_faces) > 0:
                    registered_embeddings_matrix = np.stack(registered_embeddings)

                    similarities = np.dot(registered_embeddings_matrix, face.embedding) / (
                            norm(registered_embeddings_matrix, axis=1) * norm(face.embedding))

                    max_similarity_index = np.argmax(similarities)

                    if similarities[max_similarity_index] > 0.7:
                        rounded_similarity = round(similarities[max_similarity_index], 2)
                        matched_user_name = registered_user_names[max_similarity_index]
                        cv.putText(frame_copy, f"{matched_user_name}, {rounded_similarity}", (bbox[0], bbox[1] - 10),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)

                        result_folder = "verification_results"
                        os.makedirs(result_folder, exist_ok=True)
                        result_file_path = os.path.join(result_folder,
                                                        f"{matched_user_name}_{rounded_similarity}.jpg")
                        cv.imwrite(result_file_path, frame)

                    else:
                        cv.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                        cv.putText(frame_copy, "Face not verified", (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 255), 2, cv.LINE_AA)

            cv.imshow('Face_Recognition', frame_copy)

            key = cv.waitKey(1)
            if key == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def face_attendance(self):
        cap = cv.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            frame_copy = frame.copy()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            faces = self.face_analysis.get(frame_copy)

            for idx, face in enumerate(faces):
                bbox = face.bbox.astype(int)

                cv.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                if face.kps is not None:
                    landmarks = face.kps.astype(int)

                    for point in landmarks:
                        cv.circle(frame_copy, (point[0], point[1]), 2, (0, 0, 255), 2)

                    for reg_face in self.registered_faces:
                        similarity = self.calculate_similarity(reg_face["embedding"], face.embedding)
                        if similarity > 0.7:
                            cv.putText(frame_copy, f"{reg_face['user_name']} - Present", (bbox[0], bbox[3] + 20),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                            break

            cv.imshow('Face_Attendance', frame_copy)

            key = cv.waitKey(1)
            if key == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def check_for_similarity(self, user_name, new_embedding):
        # Iterate through registered faces and check for similarity with the new embedding
        for reg_face in self.registered_faces:
            similarity = self.calculate_similarity(reg_face["embedding"], new_embedding)
            if similarity > 0.7:  # Threshold for similarity
                return True
        return False

    def check_for_similarity_before_save(self, user_name, new_embedding_matrix):
        # Iterate through registered faces and check for similarity with each new embedding
        for new_embedding in new_embedding_matrix:
            if self.check_for_similarity(user_name, new_embedding):
                return True
        return False


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
