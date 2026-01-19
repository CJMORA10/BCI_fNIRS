import scipy.io
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Directorio base donde están las carpetas de los sujetos
base_dir = Path("Dataset_fNIRS")
# Directorio principal para guardar las visualizaciones
visualization_dir = Path("Visualization_MI")
os.makedirs(visualization_dir, exist_ok=True)

# Nombres de las carpetas de los sujetos
subject_dirs = [f"subject {i:02}" for i in range(1, 29)]

# Canales motores (segmento cortical alrededor de C3 y C4)
motor_channels = list(range(24, 72))

# Índices donde se almacenan los datos de Motor Imagery en el archivo cnt
# Según la descripción: cnt{1,1}, cnt{1,3}, cnt{1,5} → índices Python 0,2,4
motor_imagery_indices = [0, 2, 4]

# Iterar sobre cada sujeto
for subject in subject_dirs:
    subject_path = base_dir / subject

    if not os.path.isdir(subject_path):
        print(f"Directory not found for {subject}. Skipping...")
        continue

    cnt_file_path = os.path.join(subject_path, "cnt.mat")
    
    if not os.path.isfile(cnt_file_path):
        print(f"File cnt.mat not found for {subject}. Skipping...")
        continue

    try:
        # Cargar el archivo cnt.mat
        cnt_data = scipy.io.loadmat(cnt_file_path)['cnt']

        # Crear una subcarpeta dentro de Visualization para este sujeto
        visualization_subject_dir = os.path.join(visualization_dir, subject)
        os.makedirs(visualization_subject_dir, exist_ok=True)

        # Iterar únicamente sobre los índices que contienen motor imagery
        for idx in motor_imagery_indices:
            # Extraer la matriz de datos del segmento
            segment_data = cnt_data[0, idx]['x'][0, 0]

            # Título descriptivo para la carpeta y la gráfica
            task_type = "Motor Imagery"
            segment_name = f"Segment_{idx + 1}"

            # Crear carpeta específica para guardar las gráficas de este segmento
            segment_dir = os.path.join(
                visualization_subject_dir, 
                f"{task_type}_{segment_name}"
            )
            os.makedirs(segment_dir, exist_ok=True)

            # Graficar únicamente los canales motores
            plt.figure(figsize=(15, 10))

            num_physiological_channels = len(motor_channels) // 2

            for ch in range(num_physiological_channels):
                col_lowWL = motor_channels[ch * 2]
                col_highWL = motor_channels[ch * 2 + 1]

                # Puedes graficar ambos WL o sólo uno (por ejemplo, WL baja)
                plt.plot(segment_data[:, col_lowWL], label=f"Channel {13 + ch} (Low WL)")
                plt.plot(segment_data[:, col_highWL], label=f"Channel {13 + ch} (High WL)", linestyle='--')

            plt.title(f"{task_type} - {segment_name} ({subject}) - Motor Physiological Channels")
            plt.xlabel("Time Points")
            plt.ylabel("Amplitude (intensity)")
            plt.legend(fontsize='small', ncol=2, loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()

            # Guardar la gráfica en la carpeta del segmento
            output_path = os.path.join(segment_dir, "Motor_channels_plot.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Plot saved: {output_path}")

    except Exception as e:
        print(f"Error processing {subject}: {e}")
