import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# Inicializar mediapipe
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils

# Variables globales
cap = None
captura_guardada = None
reconocimiento_activo = False
puntos_clave_guardados = None

# Puntos clave críticos para el reconocimiento (al menos 40 puntos)
PUNTOS_CRITICOS = [
    # Ojo izquierdo: puntos clave 33 a 42
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    # Ojo derecho: puntos clave 43 a 52
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    # Nariz: puntos clave 1 a 5
    1, 2, 3, 4, 5,
    # Boca: puntos clave 61 a 72
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
    # Contorno de la cara: puntos clave 0, 17, 18, 19, 20, 21, 22, 23
    0, 17, 18, 19, 20, 21, 22, 23
]   

def encender_webcam():
    global cap
    cap = cv2.VideoCapture(0)
    actualizar_frame()

def apagar_webcam():
    global cap
    if cap:
        cap.release()
        cap = None
    lbl_video.imgtk = None
    lbl_video.configure(image='')

def guardar_captura():
    global captura_guardada, puntos_clave_guardados
    if cap:
        ret, frame = cap.read()
        if ret:
            captura_guardada = frame
            # Guardamos los puntos clave de la cara también
            puntos_clave_guardados = obtener_puntos_clave(frame)
            cv2.imwrite('captura.png', frame)
            messagebox.showinfo("Guardar Captura", "Captura guardada exitosamente.")

def obtener_puntos_clave(frame):
    # Convertir la imagen a RGB y procesarla con FaceMesh de MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        # Extraer los puntos clave de la cara
        puntos_clave = []
        for landmarks in results.multi_face_landmarks:
            for idx in PUNTOS_CRITICOS:
                lm = landmarks.landmark[idx]
                puntos_clave.append((lm.x, lm.y, lm.z))
        return puntos_clave
    return None

def comparar_captura():
    global captura_guardada, puntos_clave_guardados
    if captura_guardada is None or puntos_clave_guardados is None:
        messagebox.showwarning("Comparar Captura", "No hay captura guardada para comparar.")
        return

    if cap:
        ret, frame = cap.read()
        if ret:
            # Obtener los puntos clave del frame actual
            puntos_clave_frame = obtener_puntos_clave(frame)

            if puntos_clave_frame:
                # Comparar los puntos clave de la captura guardada con los puntos clave actuales
                similitud = comparar_puntos_clave(puntos_clave_guardados, puntos_clave_frame)
                messagebox.showinfo("Comparar Captura", f"Porcentaje de similitud: {similitud:.2f}%")

def comparar_puntos_clave(puntos_guardados, puntos_frame):
    # Compara las posiciones relativas de los puntos clave (se puede usar una métrica como la distancia euclidiana)
    if len(puntos_guardados) != len(puntos_frame):
        return 0.0

    # Calcular la distancia promedio entre los puntos clave de la captura guardada y el frame actual
    distancia_total = 0
    for p1, p2 in zip(puntos_guardados, puntos_frame):
        distancia_total += np.linalg.norm(np.array(p1) - np.array(p2))

    # Normalizar la similitud (cuanto menor la distancia, mayor la similitud)
    similitud = 100 * (1 - distancia_total / len(puntos_guardados))
    return max(0, similitud)

def iniciar_reconocimiento():
    global reconocimiento_activo
    if captura_guardada is None or puntos_clave_guardados is None:
        messagebox.showwarning("Iniciar Reconocimiento", "No hay captura guardada para comparar.")
        return
    reconocimiento_activo = True
    comparar_en_tiempo_real()

def detener_reconocimiento():
    global reconocimiento_activo
    reconocimiento_activo = False
    messagebox.showinfo("Detener Reconocimiento", "Reconocimiento detenido.")

def comparar_en_tiempo_real():
    global reconocimiento_activo
    if reconocimiento_activo and captura_guardada is not None and puntos_clave_guardados is not None:
        if cap:
            ret, frame = cap.read()
            if ret:
                # Obtener los puntos clave del frame actual
                puntos_clave_frame = obtener_puntos_clave(frame)
                
                if puntos_clave_frame:
                    # Comparar los puntos clave de la captura guardada con los puntos clave actuales
                    similitud = comparar_puntos_clave(puntos_clave_guardados, puntos_clave_frame)
                    
                    if similitud >= 95:
                        messagebox.showinfo("Reconocimiento Exitoso", "Reconocimiento exitoso")
                        reconocimiento_activo = False

    if reconocimiento_activo:
        root.after(500, comparar_en_tiempo_real)

def actualizar_frame():
    global cap
    if cap:
        ret, frame = cap.read()
        if ret:
            # Detección de rostros usando mediapipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(img_rgb)
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    # Dibujar puntos clave en el rostro
                    for idx in PUNTOS_CRITICOS:
                        lm = landmarks.landmark[idx]
                        h, w, _ = frame.shape
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), 2)

            # Convertir la imagen a formato compatible con tkinter
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_video.imgtk = imgtk
            lbl_video.configure(image=imgtk)

        lbl_video.after(10, actualizar_frame)

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Reconocimiento Facial")

# Crear un frame para los botones y colocarlo en la parte superior
frame_botones = tk.Frame(root)
frame_botones.pack(side=tk.TOP, fill=tk.X)

# Crear los botones y colocarlos dentro del frame en una sola fila
btn_encender = tk.Button(frame_botones, text="Encender Webcam", command=encender_webcam)
btn_encender.pack(side=tk.LEFT, fill=tk.X, expand=True)

btn_apagar = tk.Button(frame_botones, text="Apagar Webcam", command=apagar_webcam)
btn_apagar.pack(side=tk.LEFT, fill=tk.X, expand=True)

btn_guardar = tk.Button(frame_botones, text="Guardar Captura", command=guardar_captura)
btn_guardar.pack(side=tk.LEFT, fill=tk.X, expand=True)

btn_comparar = tk.Button(frame_botones, text="Comparar Captura", command=comparar_captura)
btn_comparar.pack(side=tk.LEFT, fill=tk.X, expand=True)

btn_reconocimiento = tk.Button(frame_botones, text="Iniciar Reconocimiento", command=iniciar_reconocimiento)
btn_reconocimiento.pack(side=tk.LEFT, fill=tk.X, expand=True)

btn_detener_reconocimiento = tk.Button(frame_botones, text="Detener Reconocimiento", command=detener_reconocimiento)
btn_detener_reconocimiento.pack(side=tk.LEFT, fill=tk.X, expand=True)

# Crear el widget de video y colocarlo debajo del frame de botones
lbl_video = tk.Label(root)
lbl_video.pack()

# Iniciar el bucle principal
root.mainloop()