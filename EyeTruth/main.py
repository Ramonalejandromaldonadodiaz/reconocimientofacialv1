import cv2
import mediapipe as mp
from deepface import DeepFace
import threading

# Inicializar Mediapipe para detección facial
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def analizar_emocion(frame, emocion_resultante):
    """
    Función que analiza las emociones en el frame y guarda el resultado.
    """
    try:
        # Análisis de la emoción usando DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Verificar si se obtuvo un resultado en forma de lista o diccionario
        if isinstance(result, list) and len(result) > 0:
            emocion_resultante[0] = result[0]['dominant_emotion']
        elif isinstance(result, dict):
            emocion_resultante[0] = result['dominant_emotion']
        else:
            emocion_resultante[0] = "No se detectó rostro"
    except Exception as e:
        emocion_resultante[0] = f"Error: {str(e)}"

# Acceso a la cámara
cap = cv2.VideoCapture(0)

print("Iniciando el reconocimiento facial. Presiona 'q' para salir.")

emocion_actual = ["No se detectó rostro"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB ya que Mediapipe trabaja en RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detección de rostro con Mediapipe
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                # Convertir los puntos en coordenadas absolutas de la imagen
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)

                # Dibujar los puntos faciales en color rojo
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Iniciar el análisis de emoción en un hilo separado para no bloquear la cámara
    threading.Thread(target=analizar_emocion, args=(frame, emocion_actual)).start()

    # Mostrar la emoción detectada en el frame
    cv2.putText(frame, f'Emoción detectada: {emocion_actual[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Mostrar la imagen en una ventana
    cv2.imshow("Sistema de Reconocimiento de Emociones", frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
