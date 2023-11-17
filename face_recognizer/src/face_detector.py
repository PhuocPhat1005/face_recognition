from insightface.app import FaceAnalysis
import cv2 as cv
import numpy as np
from numpy.linalg import norm
import os

# Initialize FaceAnalysis globally
face_analysis = FaceAnalysis(
    name="buffalo_l", 
    allow_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"])
    
face_analysis.prepare(ctx_id=0, det_size=(640, 480))

def load_embeddings_from_folder(folder_path):
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

def face_register(user_name, registered_faces):
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    for index in range(3):
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            faces = face_analysis.get(frame)

            for idx, face in enumerate(faces):
                bbox = face.bbox.astype(int)

                cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                if face.kps is not None:
                    landmarks = face.kps.astype(int)

                    for point in landmarks:
                        cv.circle(frame, (point[0], point[1]), 2, (0, 0, 255), 2)
            if index == 0:
                cv.putText(frame, "Frontal_Face", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            elif index == 1:
                cv.putText(frame, "Left_Face", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            elif index == 2:
                cv.putText(frame, "Right_Face", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv.imshow('Face-Register', frame)

            key = cv.waitKey(1)
            if key == ord('c'):
                # Check if the user is already registered
                for reg_face in registered_faces:
                    if np.array_equal(reg_face["embedding"], faces[0].embedding):
                        print("This person is already registered.")
                        break
                else:
                    # Lưu frame và thông tin đăng ký vào danh sách
                    registered_faces.append({"user_name": user_name, "embedding": faces[0].embedding})
                    np.save(f"register/embeddings/{user_name}/{user_name}_{index + 1}.npy", faces[0].embedding)
                    cv.imwrite(f"register/pics/{user_name}/{user_name}_{index + 1}.jpg", frame)
                    print(f"Face and landmarks of {user_name} captured and saved as {user_name}_{index + 1}.jpg")
                    break  # Exit the loop after successful registration

            elif key == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

def calculate_similarity(embeddings1, embeddings2):
    # You can use various methods to calculate similarity, here I'm using dot product
    similarity = np.dot(embeddings1, embeddings2) / (norm(embeddings1) * norm(embeddings2))
    return similarity

def face_verify(registered_faces):
    cap = cv.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        faces = face_analysis.get(frame)

        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)

            cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            if face.kps is not None:
                landmarks = face.kps.astype(int)

                for point in landmarks:
                    cv.circle(frame, (point[0], point[1]), 2, (0, 0, 255), 2)

            for registered_face in registered_faces:
                # Xác minh khuôn mặt
                similarity = calculate_similarity(registered_face["embedding"], face.embedding)
                # Kiểm tra xem độ tương đồng có vượt qua ngưỡng không
                if similarity > 0.7:  # Giả sử ngưỡng là 0.7
                    rounded_similarity = round(similarity, 2)
                    cv.putText(frame, f"{registered_face['user_name']}, {rounded_similarity}", (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
                    
                    # Save the result to a folder
                    result_folder = "verification_results"
                    os.makedirs(result_folder, exist_ok=True)
                    result_file_path = os.path.join(result_folder, f"{registered_face['user_name']}_{rounded_similarity}.jpg")
                    cv.imwrite(result_file_path, frame)
                    
                else:
                    cv.putText(frame, "Face not verified", (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

        cv.imshow('Detected Faces and Landmarks', frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def main():
    # Load registered faces from the "register/embeddings" folder
    registered_faces = load_embeddings_from_folder("register/embeddings")

    while True:
        user_name = input("Enter your name (or 'q' to quit): ")
        if user_name.lower() == 'q':
            break
        choose = input("Choose an action (1: Register, 2: Verify): ")
        choose = int(choose)
        
        if choose == 1:
            face_register(user_name, registered_faces)
        elif choose == 2:
            face_verify(registered_faces)

if __name__ == "__main__":
    main()
