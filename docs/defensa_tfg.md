# Guía de defensa — TFG Seguimiento de Ratones con Visión por Computador

---

## 1. Las 7 posturas: cómo las detecta el sistema

El sistema usa **lógica híbrida**: YOLO clasifica la postura, pero el código la corrige o refina frame a frame con velocidad, keypoints y geometría espacial. Las 7 etiquetas finales son:

| Etiqueta final | Cómo se decide |
|---|---|
| `rat_rearing` | YOLO lo predice Y el bounding box tiene aspect ratio `alto/ancho ≥ 0.70`. Si el ratio es menor → se reenvía a walking/immobile. |
| `rat_climbing` | YOLO lo predice Y el bbox sobresale de la zona interior calibrada. O bien YOLO dice `rat_horizontal` y el snout está fuera del borde interior. |
| `rat_head_dipping` | El keypoint del snout está a menos de `0.9 × radio` de algún agujero calibrado. YOLO también puede activarlo directamente si YOLO lo predice (por si el snout queda tapado). |
| `rat_grooming` | YOLO lo predice directamente; el clasificador lo acepta tal cual. |
| `walking` | Velocidad del centroide `≥ 0.35` (escala normalizada). |
| `immobile` | Velocidad `≤ 0.15`. |
| `sniffing_walking` / `sniffing_immobile` | La clase `rat_horizontal` de YOLO, cuando el snout está cerca de una pared interior (a menos de 30 px del borde interior), se convierte en `sniffing`. Luego se subdivide según velocidad. |

**La clase `rat_horizontal` de YOLO es ruido puro**: el modelo la usa cuando el ratón está tumbado, pero es la postura más ambigua. El código **nunca la muestra**; la desambigua en climbing / sniffing / walking / immobile usando posición del snout y velocidad.

### Estabilizador temporal (`_LabelStabilizer`)
Para evitar parpadeo entre etiquetas, una etiqueta nueva solo reemplaza a la actual si aparece **8 frames consecutivos** (≈ 0.5 s a 15 fps). Las posturas puntuales como rearing o head_dipping necesitan solo **3 frames** para reaccionar más rápido.

### Parámetros que puedes ajustar en `behavior_classifier.py`

```python
WALK_SPEED_THRESHOLD  = 0.35   # sube → menos frames clasificados como walking
STILL_SPEED_THRESHOLD = 0.15   # baja → más frames como immobile
REARING_ASPECT_RATIO  = 0.70   # sube → más exigente para confirmar rearing
SNOUT_CONF_HIGH       = 0.55   # confianza mínima del keypoint para anular rearing/climbing
```

---

## 2. GPU vs CPU: detección y uso

### Dónde se detecta el hardware

```python
# scripts/config/config.py  línea 23
import torch
_DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"
```

Al arrancar la app, `torch.cuda.is_available()` comprueba si hay una GPU CUDA instalada en el sistema. Si la hay, el dispositivo por defecto es `"0"` (primera GPU); si no, `"cpu"`.

### En la CLI — el usuario elige

```python
# scripts/main_model.py  línea 32-55
def _select_device():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        choice = input("[?] Usar GPU (Enter) o CPU? [G/c]: ")
        if choice == "c":
            detect_cfg.device = "cpu"
        else:
            detect_cfg.device = "0"
    else:
        detect_cfg.device = "cpu"
```

### En la GUI — automático

```python
# gui/controllers/detect_worker.py  línea 155
device = "0" if torch.cuda.is_available() else "cpu"
```

### Cómo se usa en inferencia

```python
# scripts/detection/detector.py  línea 256-259
results = self.model.predict(
    source=source_yolo, stream=True,
    conf=self.cfg.conf_threshold, device=self.cfg.device, iou=0.5
)
```

Solo cambia el parámetro `device`. **Los kernels CUDA no están escritos a mano**: Ultralytics/PyTorch los ejecuta internamente cuando detecta `device="0"`. El modelo se mueve a la VRAM de la GPU y la inferencia se paraleliza en miles de núcleos CUDA. Desde el código, es literalmente una cadena de texto distinta.

### Para el entrenamiento — batch size dinámico por VRAM

```python
# scripts/detection/trainer.py  línea 30-35
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
if gpu_mem >= 8:  return 8
if gpu_mem >= 4:  return 4
return 2
```

---

## 3. Canvas de trayectoria

```python
# scripts/utils/stats_generator.py  línea 166-203
img = np.full((700, 700, 3), 255, dtype=np.uint8)  # lienzo blanco 700×700
```

**Cómo funciona paso a paso:**

1. Se crea una imagen numpy de fondo blanco (`np.full(..., 255)`).
2. Se dibuja la cuadrícula con `cv2.line` en gris muy claro.
3. Se dibuja el borde de la caja con `cv2.rectangle`.
4. Se leen las coordenadas del snout de cada fila del CSV (`snout_x`, `snout_y`).
5. Las coordenadas del video (ej. 1920×1080 px) se **escalan al canvas de 700×700** usando los límites de calibración.
6. Se conecta cada punto con el anterior con `cv2.line(img, prev_pt, pt, COLOR_TRACK, 1)`.

**¿Por qué el snout y no el centroide del bbox?** El snout es el punto más representativo del comportamiento exploratorio: refleja a dónde mira y huele el ratón, no solo dónde está su cuerpo.

### La escala (video → canvas)

```python
x_scale = area / (x_max_calibrado - x_min_calibrado)
cx = int(margen + (snout_x - x_min) * x_scale)
```

Si no hay calibración, usa el rango real de coordenadas del snout en ese video.

---

## 4. Mapa de calor

```python
# scripts/utils/stats_generator.py  línea 243-260
accum = np.zeros((700, 700), dtype=np.float32)   # matriz de ceros

for row in self.rows:
    cx, cy = self._to_canvas(snout_x, snout_y, ...)
    accum[cy, cx] += 1.0                          # +1 en cada pixel visitado

accum = cv2.GaussianBlur(accum, (0, 0), sigmaX=12)  # suavizar manchas

if accum.max() > 0:
    accum = accum / accum.max()                   # normalizar a [0,1]

img = cv2.applyColorMap((accum * 255).astype(np.uint8), cv2.COLORMAP_JET)
```

**Paso a paso:**

1. **Matriz de acumulación**: `np.zeros` crea una matriz float de ceros del tamaño del canvas. Cada vez que el snout está en un pixel, ese pixel suma 1. Un pixel con valor 300 significa que el ratón pasó 300 frames por ahí.
2. **Gaussian blur**: convierte los puntos discretos en manchas suaves. Sin esto, el mapa serían píxeles individuales.
3. **Normalización**: divide todo por el máximo, para que el pixel más visitado sea siempre 1.0 independientemente del vídeo.
4. **COLORMAP_JET**: azul = zona poco visitada, verde = medio, rojo = hotspot donde el ratón pasó más tiempo.

**Por qué no se hace con tiempos de inmovilidad**: en la primera implementación se coloreaba por segundos parado, pero el ratón se mueve la mayor parte del tiempo → el mapa salía todo azul sin variación. La acumulación de presencia es más informativa.

---

## 5. Estadísticas y Excel

> **Importante**: YOLO NO genera el Excel del experimento de forma automática. YOLO genera métricas de su propio entrenamiento (precision, recall, mAP). El Excel del experimento lo generamos nosotros.

### El CSV que sí genera nuestro código

Durante la detección, `CsvOutput` escribe en un CSV propio fila a fila:
- `frame`, `time_s`, `yolo_label`, `final_label`
- `x1, y1, x2, y2` (bounding box)
- `snout_x, snout_y` (keypoint snout)
- `tail_x, tail_y` (keypoint cola)
- `speed` (velocidad del centroide)

### El Excel lo genera `StatsGenerator` leyendo ese CSV

```python
StatsGenerator(csv_path, coords_json=...).generate(output_dir)
```

Genera tres archivos:
- `trajectory_*.png` — trayectoria del snout
- `heatmap_*.png` — mapa de calor
- `stats_*.xlsx` — dos hojas: **Comportamiento** (tiempo por etiqueta) y **Métricas OFT**

### Métricas OFT que calcula

- Distancia total recorrida (en metros si hay calibración, en píxeles si no)
- Velocidad media
- Head-dips: número de bouts, frecuencia por minuto, latencia al primero, habituación por cuartos
- Sniffing: % total, en movimiento vs parado, eficiencia exploratoria
- Thigmotaxis (climbing), zonas central vs periférica
- Grooming, inmovilidad, transiciones conductuales

---

## 6. Calibración: cómo funciona el clic del ratón

```python
# scripts/calibration/calibrator.py  línea 66-86

cv2.setMouseCallback(WIN_NAME, self._click_event)

def _click_event(self, event, x, y, flags, params):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    ox, oy = self._to_orig(x, y)   # convertir pantalla → frame original
    if len(self.exterior) < 2:
        self.exterior.append([ox, oy])
    elif len(self.interior) < 2:
        self.interior.append([ox, oy])
    elif len(self.holes) < 4:
        self.holes.append([ox, oy])
    self._refresh()
```

**El problema que resuelve la escala**: el vídeo puede ser 1920×1080 pero se muestra en una ventana de 1000 px de ancho. Si guardáramos las coordenadas del click en pantalla, estarían en el espacio reducido y no coincidirían con los frames del vídeo. Por eso:

```python
self.scale_x = w_orig / DISPLAY_WIDTH   # ej. 1920/1000 = 1.92
ox = int(round(x_pantalla * self.scale_x))  # se guarda en coordenadas reales
```

Todo se guarda en `coords.json`. Durante la detección, `SpatialAnalyzer` carga ese JSON y comprueba frame a frame si el snout está dentro del radio de un agujero, cerca de una pared, etc.

---

## 7. Conceptos clave para defender

### ¿Qué es YOLO?

YOLO (*You Only Look Once*) es una red neuronal convolucional (CNN) de detección de objetos en tiempo real. La idea clave es que **procesa la imagen entera en un solo paso** (a diferencia de métodos más lentos que proponen regiones primero).

- La imagen se divide en una cuadrícula; cada celda predice bounding boxes y la probabilidad de que contenga un objeto.
- **YOLO Pose** añade la predicción de **keypoints** (puntos clave) sobre el cuerpo detectado. En nuestro modelo tenemos 3: snout (hocico), spine (lomo), tail (cola).
- Nosotros **fine-tuneamos** (reentrenamos) un modelo base de YOLO con nuestro dataset de ratones.

### Métricas que aparecen durante el entrenamiento

| Métrica | Qué significa |
|---|---|
| **Precision** | De todo lo que el modelo dice que es un ratón, ¿cuánto acierta? Alta precision = pocos falsos positivos. |
| **Recall** | De todos los ratones reales, ¿cuántos detecta? Alto recall = pocos falsos negativos. |
| **mAP50** | *mean Average Precision* con IoU ≥ 0.50. IoU = solapamiento entre bbox predicho y real. Si el bbox predicho solapa más del 50% con el real, cuenta como correcto. Métrica principal de calidad. |
| **mAP50-95** | Igual pero promediando varios umbrales de IoU (0.5 a 0.95). Más exigente. |
| **box_loss** | Error en la posición del bounding box. |
| **cls_loss** | Error en la clasificación de la postura (¿es rearing o grooming?). |
| **pose_loss / kobj_loss** | Error en la posición de los keypoints y en si el keypoint existe. |

Las pérdidas (loss) deben bajar con las épocas. mAP debe subir. Si mAP50 sube pero mAP50-95 se estanca, el modelo detecta bien pero con bboxes poco precisos.

### IoU (*Intersection over Union*)

```
IoU = área(intersección) / área(unión)
```

Es la medida de solapamiento entre el bbox predicho y el real. IoU = 1 → perfecta coincidencia. IoU = 0 → no se solapan. En inferencia usamos `iou=0.5` para suprimir detecciones duplicadas (NMS, *Non-Maximum Suppression*).

### ¿Qué hace cada librería?

| Librería | Para qué la usamos |
|---|---|
| **Ultralytics** | API de alto nivel para YOLO: `YOLO("modelo.pt")`, `.train()`, `.predict()`. Abstrae el entrenamiento completo. |
| **PyTorch** (`torch`) | Motor de deep learning debajo de Ultralytics. Mueve los tensores a GPU, hace backpropagation, detecta CUDA. |
| **NumPy** (`np`) | Arrays numéricos: la imagen es un `np.ndarray`, las coordenadas son arrays, la matriz de acumulación del heatmap es `np.zeros`. |
| **OpenCV** (`cv2`) | Leer/escribir vídeo, dibujar líneas/rectángulos/círculos, aplicar colormaps, capturar eventos del ratón, redimensionar imágenes. |

### ¿Por qué `stream=True` en `model.predict()`?

```python
results = model.predict(source=video, stream=True, ...)
for res in results:   # iterador, no lista
    ...
```

Sin `stream=True`, YOLO carga todos los frames en memoria antes de devolver resultados. Con `stream=True` devuelve un **generador**: procesa y devuelve un frame, luego el siguiente. Imprescindible para vídeos largos o cámara en vivo.

---

## 8. Posibles preguntas del tribunal

**P: ¿Por qué no usáis DeepLabCut o SLEAP en vez de YOLO?**
R: YOLO Pose permite detección + pose estimation en un solo modelo, en tiempo real. DeepLabCut y SLEAP son más precisos para keypoints pero mucho más lentos y requieren más anotación.

**P: ¿Cómo garantizáis que el modelo generaliza a nuevos vídeos?**
R: Con la división train/val/test del dataset. Además, la lógica híbrida reduce la dependencia del modelo: aunque YOLO falle en un frame puntual, el estabilizador temporal y las reglas de velocidad/geometría mantienen la clasificación coherente.

**P: ¿Qué pasa si no hay calibración?**
R: El sistema detecta que `coords.json` no existe o no es compatible con las dimensiones del vídeo. En ese caso desactiva head_dipping, climbing y sniffing (que requieren geometría espacial), y solo usa velocidad + YOLO para el resto de posturas. El vídeo se procesa igualmente.

**P: ¿Por qué el snout y no el centroide del bbox para la trayectoria?**
R: El centroide del bbox marca el centro del cuerpo. El snout marca a dónde va la nariz, que es el punto de exploración real. En experimentos OFT/Holeboard, lo que importa es dónde huele el ratón, no dónde está su lomo.

**P: ¿Qué es un bout?**
R: Una racha continua de frames con la misma etiqueta. Si el ratón hace grooming durante 2 s, para 0.5 s y vuelve a hacerlo 1 s, son 2 bouts de grooming. El número de bouts mide la frecuencia de aparición de un comportamiento, no solo su duración total.

**P: ¿Generáis el Excel con YOLO?**
R: No. YOLO genera sus propias métricas de entrenamiento (en `runs/train/`). El Excel del experimento lo generamos nosotros con `StatsGenerator`, que lee el CSV que escribimos frame a frame durante la inferencia.

**P: ¿Por qué va mejor con CPU en vídeo grabado que con cámara en vivo?**
R: Ver sección 9.

---

## 9. CPU vs cámara en vivo — por qué el vídeo grabado es más tolerante

No es que CPU sea más rápido procesando — es que el vídeo grabado **no tiene restricción de tiempo real**.

- **Vídeo grabado**: OpenCV lee el archivo a la velocidad que el sistema pueda. Si la inferencia tarda 100 ms por frame, simplemente procesa un frame cada 100 ms y el vídeo de salida queda correcto igualmente. No hay presión.
- **Cámara en vivo**: los frames llegan a 30 fps fijos, es decir, cada 33 ms. Si la inferencia tarda 100 ms, para cuando terminas un frame ya han llegado 3 más que el buffer ha descartado. El resultado es imagen saltando o retardo creciente que no se recupera.

Para cámara en vivo, la inferencia necesita ser más rápida que el intervalo entre frames (< 33 ms a 30 fps). Con CPU, YOLO suele tardar entre 80 y 200 ms por frame según la máquina, lo que es insuficiente para tiempo real fluido. Con GPU, ese tiempo baja a 10-30 ms.

---

## 10. Crear el ejecutable (.exe) con PyInstaller

PyInstaller empaqueta el proyecto y todas sus dependencias en una carpeta que el usuario final puede ejecutar sin tener Python instalado.

### Instalación (con el entorno del proyecto activado)

```
python -m pip install pyinstaller
```

No está en `requirements.txt` porque es una herramienta de desarrollo, no una dependencia del sistema.

### Comando para generar el ejecutable

Desde la raíz del proyecto:

```
pyinstaller --onedir --windowed --name MiceTracker --add-data "models;models" gui/app.py
```

| Flag | Para qué sirve |
|---|---|
| `--onedir` | Genera una carpeta con todo dentro (no un único .exe). Imprescindible con torch: arranca en segundos en vez de 30-60 s. |
| `--windowed` | No abre la consola negra al ejecutar (modo GUI). |
| `--name MiceTracker` | Nombre del ejecutable generado. |
| `--add-data "models;models"` | Incluye la carpeta de modelos (`.pt`) que PyInstaller no detecta automáticamente. |

### Resultado

Se crea `dist/MiceTracker/`. Se comprime esa carpeta entera en un zip. El usuario la descomprime y hace doble click en `MiceTracker.exe`. No necesita Python, pip ni nada.

---

## 11. CUDA — ¿cuándo detecta GPU?

`torch.cuda.is_available()` devuelve `True` si se cumplen **las tres** condiciones:

1. Hay una GPU NVIDIA en el sistema
2. Los drivers CUDA de NVIDIA están instalados
3. PyTorch fue compilado con soporte CUDA

En la práctica: **NVIDIA + drivers = GPU**. AMD o Intel integrada → siempre CPU.

### Importante: el requirements.txt instala la versión CPU de torch

```
torch==2.7.1+cpu
```

Con esta versión, `cuda.is_available()` devuelve siempre `False` aunque el ordenador tenga una NVIDIA. Para activar GPU habría que reinstalar:

```
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

Esto es intencional para el entorno de desarrollo: la versión CPU funciona en cualquier máquina sin depender de drivers CUDA.

---

## 12. Esquema de archivos — qué hace cada módulo de `scripts/`

```
scripts/
│
├── main_model.py
│     Menú principal de la CLI. Detecta GPU/CPU al arrancar y orquesta
│     todas las opciones: calibrar, entrenar, detectar, generar stats.
│     Es el punto de entrada si se usa el proyecto sin GUI.
│
├── config/
│   ├── config.py
│   │     Configuración centralizada. Define todas las rutas del proyecto
│   │     (modelo, vídeo, salidas, coords.json) y los parámetros de
│   │     entrenamiento (epochs, imgsz, batch) y detección (conf, iou).
│   │     Todo el código importa de aquí — es la única fuente de verdad.
│   │
│   └── interfaces.py
│         Clase base abstracta BaseModule con el método run() que deben
│         implementar todos los módulos (Detector, Trainer, Calibrator...).
│         Garantiza que todos tienen la misma interfaz.
│
├── detection/
│   ├── detector.py
│   │     Módulo principal de inferencia. Carga el modelo YOLO, procesa
│   │     el vídeo frame a frame, extrae keypoints (snout, spine, tail),
│   │     calcula la velocidad del centroide y delega la clasificación
│   │     final en BehaviorClassifier. Escribe el vídeo anotado y el CSV.
│   │
│   └── trainer.py
│         Lanza el fine-tuning de YOLO con el dataset propio. Estima el
│         batch size óptimo según la VRAM disponible y copia el mejor
│         modelo entrenado (best.pt) a models/yolo_ratas.pt.
│
├── behavior/
│   └── behavior_classifier.py
│         Lógica híbrida que corrige y refina la etiqueta bruta de YOLO.
│         Aplica reglas de velocidad, aspect ratio del bbox, posición del
│         snout y geometría espacial para producir las 7 etiquetas finales.
│         Incluye el estabilizador temporal que evita el parpadeo.
│
├── spatial/
│   └── spatial.py
│         Analiza la posición del ratón respecto a la caja calibrada.
│         Carga coords.json y expone métodos concretos:
│           · check_dipping()       → snout dentro del radio de un agujero
│           · snout_in_wall_zone()  → snout más allá del borde interior
│           · check_sniffing_wall() → snout cerca de la pared interior
│           · bbox_entirely_inside()→ bbox completamente dentro del suelo
│           · draw_zones()          → dibuja los rectángulos en el frame
│
├── calibration/
│   ├── calibrator.py
│   │     Calibrador interactivo sobre el primer frame de un vídeo.
│   │     Abre una ventana OpenCV, captura clics del ratón y guarda
│   │     en coords.json: borde exterior, borde interior y 4 agujeros.
│   │     Convierte coordenadas de pantalla a coordenadas reales del frame.
│   │
│   └── calibrator_image.py
│         Igual que calibrator.py pero acepta una imagen estática en vez
│         de un vídeo. Útil si no se tiene el vídeo pero sí un frame.
│
├── output/
│   └── writers.py
│         Dos clases de escritura que usa el Detector:
│           · VideoOutput — escribe el vídeo anotado (y opcionalmente
│             una copia limpia sin anotaciones) con cv2.VideoWriter.
│           · CsvOutput   — escribe fila a fila el CSV con frame, tiempo,
│             etiqueta YOLO, etiqueta final, bbox, keypoints y velocidad.
│
└── utils/
    └── stats_generator.py
          Lee el CSV generado por el Detector y produce tres salidas:
            · trajectory_*.png  — trayectoria del snout sobre canvas
            · heatmap_*.png     — mapa de calor de presencia
            · stats_*.xlsx      — Excel con hoja de comportamiento
                                  y hoja de métricas OFT (head-dips,
                                  distancia, zonas, grooming, etc.)
```

Los `__init__.py` están vacíos — solo marcan cada carpeta como paquete Python para que los imports funcionen.
