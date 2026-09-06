# Orden de código y teoría — Guía práctica del pipeline

> **A quién va dirigido:** alguien con Python base + pandas que aún no conoce OpenCV, PyTorch ni YOLO.
> **Cómo usarlo:** lee primero el bloque 0 (puente teoría–código), después los archivos en el orden de la tabla, haz el ejercicio de cada grupo antes de pasar al siguiente, y usa la checklist de autoevaluación al final de la semana.

---

## 0. Puente teoría–código

Estos bloques no son un curso de teoría — para eso ya tienes tus recursos externos. Son anclas: cada concepto teórico se conecta con la línea de código exacta donde aparece, para que cuando leas la teoría sepas a qué parte del proyecto corresponde, y cuando leas el código sepas por qué está escrito así.

---

### Bloque A — De píxeles a "hay una rata aquí": CNN + YOLO

**Concepto:** una CNN es una pila de filtros que se deslizan sobre la imagen. Los primeros aprenden a detectar bordes y texturas, los últimos aprenden formas complejas (cabeza de rata, cola, postura). YOLO añade encima una **cabeza de detección** que divide la imagen en una cuadrícula y predice, para cada celda, si hay un objeto, dónde está exactamente (bounding box) y a qué clase pertenece. Todo esto ocurre en un solo paso hacia adelante por la red — de ahí "You Only Look Once".

**En el código — `scripts/detection/detector.py`:**

```python
# _setup() — carga la arquitectura entera (CNN + cabeza de detección)
self.model = YOLO(str(paths.yolo_model))

# run() — un solo forward pass por frame, devuelve todas las predicciones
results = self.model.predict(
    source=str(paths.video_source), stream=True,
    conf=self.cfg.conf_threshold, device=self.cfg.device, iou=0.5
)

# Para cada frame, la predicción con más confianza
confs   = res.boxes.conf.cpu().numpy()   # probabilidades de cada detección
best    = int(np.argmax(confs))          # la de mayor confianza
rat_box = res.boxes.xyxy[best].cpu().numpy()  # [x1, y1, x2, y2] en píxeles
```

`res.boxes.conf` es el número que la red asigna a cada caja como "cuánto confío en que aquí hay una rata de esta clase". No es un porcentaje de certeza absoluta — es la salida de la última capa multiplicada por la probabilidad de objeto. Cuando ese número baja de `conf_threshold`, YOLO ya ni te devuelve esa caja.

---

### Bloque B — Por qué 0.18 y no 0.5: conf_threshold

**Concepto:** la red produce una puntuación continua para cada posible detección. `conf_threshold` es el corte: por debajo se descarta, por encima se conserva. Un umbral alto (0.5) exige mucha seguridad → pocas detecciones, pocas falsas alarmas, pero se pierden comportamientos raros. Un umbral bajo (0.18) acepta detecciones menos seguras → más detecciones, posibles falsos positivos, pero los comportamientos minoritarios sobreviven. El spatial logic de abajo los filtra si no tienen sentido geométrico.

**En el código — `scripts/config/config.py` + `scripts/detection/detector.py`:**

```python
# config.py — el único sitio donde se cambia
@dataclass
class DetectParams:
    conf_threshold: float = 0.18   # bajado de 0.25 en Phase 6

# detector.py — donde se aplica
results = self.model.predict(..., conf=self.cfg.conf_threshold, ...)
```

En la Fase 6 del proyecto se bajó de 0.25 a 0.18 porque `rat_grooming` y `rat_rearing` (clases minoritarias) se estaban descartando sistemáticamente. Eso no es casualidad: las clases con pocos ejemplos de entrenamiento producen puntuaciones más bajas en promedio, y un umbral estándar las elimina antes de que lleguen a la lógica espacial.

---

### Bloque C — Los tres puntos del cuerpo: estimación de pose y keypoints

**Concepto:** YOLOv8-Pose añade a la cabeza de detección estándar una segunda rama que predice la posición de **keypoints** (puntos del esqueleto). Para cada detección devuelve K puntos, cada uno con coordenadas `(x, y)` y una **confianza de visibilidad** — qué tan segura está la red de que ese punto es visible en la imagen. Un keypoint con confianza baja o coordenadas `(0, 0)` significa que la red no lo pudo estimar.

**En el código — `scripts/detection/detector.py`:**

```python
# kpt_shape=[3, 3]: 3 keypoints, cada uno con (x, y, conf)
# Índices definidos como constantes al principio del archivo:
KP_SNOUT: int = 0   # hocico
KP_SPINE: int = 1   # lomo / columna
KP_TAIL:  int = 2   # base de la cola

# Para extraer el snout con filtro de confianza:
snout = kps_xy[detection_idx][KP_SNOUT].cpu().numpy()   # coordenadas (x, y)
conf  = float(kps_conf[detection_idx][KP_SNOUT].cpu())  # confianza [0, 1]
if conf < 0.3:
    return None   # keypoint no fiable: descartar

# YOLO representa keypoints no visibles como (0, 0)
if snout[0] < 1.0 and snout[1] < 1.0:
    return None
```

El hocico (`snout_kp`) es el keypoint más importante del pipeline: es el que decide si hay head_dipping (`check_dipping(snout_kp)`) y si hay sniffing de pared (`check_sniffing_wall(snout_kp)`). La cola (`tail_kp`) solo se usa para la trayectoria en el Excel — es el punto más estable del cuerpo durante la locomoción.

---

### Bloque D — Por qué no empezamos de cero: transfer learning

**Concepto:** entrenar una CNN desde cero para detectar ratas requeriría cientos de miles de imágenes. Transfer learning parte de un modelo ya entrenado en un dataset enorme (COCO, 80 clases, millones de imágenes) que ya "sabe" detectar bordes, formas y siluetas de animales. El fine-tuning solo ajusta los pesos finales para que aprenda las 5 clases de postura de rata. Con 248 imágenes originales esto es posible; sin transfer learning, no lo sería.

**En el código — `scripts/config/config.py` + `scripts/detection/trainer.py`:**

```python
# config.py — modelo base pre-entrenado (pesos de Ultralytics/COCO)
base_yolo_model: Path = models_dir / "yolov8s-pose.pt"

# trainer.py — fine-tuning, no entrenamiento desde cero
self.model = YOLO(self.cfg.base_model)   # carga los pesos pre-entrenados
self.model.train(
    data=str(paths.data_yaml),
    epochs=100,        # pocos epochs porque la red ya sabe mucho
    ...
)
```

El augmentation geométrico (rotaciones, espejos, escalado) es especialmente importante en fine-tuning con datasets pequeños: sin él, la red memoriza las 248 imágenes exactas en pocos epochs. Con él, ve ~1100 variaciones diferentes y generaliza mejor.

---

### Bloque E — IoU y NMS: por qué YOLO no devuelve 50 cajas para la misma rata

**Concepto:** YOLO genera cientos de candidatos de caja por frame (uno por celda de la cuadrícula). La mayoría se solapan sobre el mismo objeto. **Non-Maximum Suppression (NMS)** los filtra: calcula el **IoU** (Intersection over Union = área de solapamiento / área de unión) entre cajas y descarta las que se solapan demasiado con una de mayor confianza. El parámetro `iou=0.5` significa: si dos cajas comparten más del 50 % de su área, se queda solo la más confiada.

**En el código — `scripts/detection/detector.py`:**

```python
# YOLO aplica NMS internamente con el umbral especificado
results = self.model.predict(..., iou=0.5)

# Después de NMS puede quedar más de una caja si hay varios animales
# (o detecciones muy separadas). El código se queda con la de más confianza:
confs = res.boxes.conf.cpu().numpy()
best  = int(np.argmax(confs))   # índice de la detección más segura
```

En este experimento solo hay una rata por vídeo, así que NMS prácticamente siempre deja una sola caja. La línea `np.argmax(confs)` es una salvaguarda para el caso en que NMS no sea suficiente o aparezca un reflejo.

---

### Bloque F — LSTM: qué intentaba hacer y por qué falla aquí

**Concepto:** una LSTM es un tipo de red recurrente que procesa **secuencias**. Mantiene un vector de memoria (estado oculto `h`) que se actualiza en cada paso de tiempo. La idea era: una imagen sola no distingue walking de immobile (el bbox es idéntico), pero 30 frames seguidos revelan si hay desplazamiento. La LSTM leería esos 30 frames y clasificaría el comportamiento usando contexto temporal.

**En el código — `adminScripts/rnn/model.py`:**

```python
# Entrada: (batch, 30 frames, 5 features por frame)
# Las 5 features por frame son: cx, cy, w, h, speed  (ver dataset.py)
out, _ = self.lstm(x, (h0, c0))

# Solo el último paso de tiempo: ha "visto" los 30 frames anteriores
out = out[:, -1, :]   # (batch, hidden_size)
out = self.fc(out)    # (batch, num_classes)
```

**Por qué no funcionó** (problema conceptual, no técnico): los datos de entrenamiento venían de los CSVs generados por `detector.py`. Eso significa que la LSTM aprendía los patrones de error de YOLO, no el comportamiento real de la rata. Si YOLO etiquetaba mal 3 frames seguidos, la LSTM aprendía que eso era la señal correcta. Este problema se llama **supervisión circular** y no tiene solución sin un etiquetado manual independiente.

La solución que se adoptó (`_LabelStabilizer`) hace exactamente lo que necesitaba la LSTM (suavizado temporal), pero de forma determinista: requiere que una etiqueta aparezca 8 frames seguidos antes de aceptarla. Sin entrenamiento, sin datos, sin supervisión circular.

---

### Bloque G — Nuevas funcionalidades: camara en vivo, cuadricula, y mapa de calor

---

#### G.1 Camara en vivo (Opcion 3 del menu)

**Concepto:** YOLO acepta como fuente no solo rutas a archivos sino tambien indices de camara
(0 = camara principal del sistema). Cuando se pasa `source=0` a `model.predict()`, Ultralytics
abre la camara directamente y devuelve frames en el mismo formato que con un video.

**En el codigo — `scripts/main_model.py` + `scripts/detection/detector.py`:**

```python
# main_model.py — opcion 3: capturar un frame para calibracion, luego detectar
cap = cv2.VideoCapture(0)    # 0 = camara por defecto del sistema
ok, frame = cap.read()       # un solo frame para mostrar en el calibrador
cap.release()

# Una vez calibrado, RatDetector arranca con camera_index=0
detector = RatDetector(detect_cfg, show_preview=True, camera_index=0)
detector.run()

# detector.py — dentro de run()
source_yolo = self.camera_index   # puede ser 0 (camara) o str (ruta de video)
results = self.model.predict(source=source_yolo, stream=True, ...)
```

El flujo completo:
1. `main_model.py` abre la camara, captura un solo frame y lo cierra.
2. `ImageCalibrator` muestra ese frame y el usuario marca los bordes.
3. `RatDetector.run()` vuelve a abrir la camara (via YOLO) y procesa en tiempo real.
4. El usuario pulsa Q en la ventana de preview para parar.
5. El CSV y las estadisticas se generan al terminar, igual que con video.

La diferencia clave respecto al modo video: el detector tiene `show_preview=True`
porque de lo contrario no habria forma de ver lo que esta pasando ni de parar.

---

#### G.2 Cuadricula en el calibrador

**Concepto:** OpenCV no tiene una funcion de cuadricula nativa. Se dibuja linea a linea
usando `cv2.line()` sobre una copia de la imagen y luego se fusiona con `cv2.addWeighted()`
para que sea semitransparente. El alpha bajo (0.25) hace que las lineas sean apenas visibles
sin tapar al raton.

**En el codigo — `scripts/calibration/calibrator_image.py`:**

```python
def _draw_grid(self, img: np.ndarray, divisions: int = 8) -> None:
    h, w = img.shape[:2]
    overlay = img.copy()                         # copia limpia para dibujar encima
    for i in range(1, divisions):
        x = int(w * i / divisions)
        cv2.line(overlay, (x, 0), (x, h), GRAY, 1)   # lineas verticales
    for j in range(1, divisions):
        y = int(h * j / divisions)
        cv2.line(overlay, (0, y), (w, y), GRAY, 1)   # lineas horizontales

    # addWeighted mezcla overlay (con la cuadricula) con img (sin ella):
    #   img_final = 0.25 * overlay + 0.75 * img
    # -> la cuadricula solo aporta el 25% de intensidad, muy sutil
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
```

La cuadricula se dibuja sobre la imagen ya escalada a pantalla (no sobre la original),
para que las lineas tengan siempre 1 pixel de grosor sin importar la resolucion del video.

---

#### G.3 Mapa de calor dinamico

**Concepto:** el mapa de calor usa exactamente la misma base que la trayectoria
(un canvas `np.zeros` con `cv2.line()`), pero en lugar de un color fijo para todos los
segmentos, el color de cada segmento depende de cuanto tiempo lleva el raton parado en
esa zona.

**Algoritmo — `scripts/utils/stats_generator.py`:**

```python
# Variables de estado que se actualizan frame a frame
dwell_frames: int = 0    # contador de frames consecutivos parado

for row in self.rows:
    speed = float(row["speed"])   # velocidad del centroide (de _SpeedTracker)

    # Si el raton se mueve: reset del contador (color frio)
    # Si el raton esta parado: incrementar el contador (hacia color caliente)
    if speed < 0.10:              # umbral de inmovilidad
        dwell_frames += 1
    else:
        dwell_frames = 0

    # heat: fraccion del tiempo maximo alcanzado [0.0, 1.0]
    # A 30 fps, _HEAT_MAX_DWELL=60 -> 2 segundos parado = color rojo completo
    heat = min(1.0, dwell_frames / 60)

    color = _heat_to_bgr(heat)          # azul -> verde -> rojo
    cv2.line(img, prev_pt, pt, color, 1)
```

**Gradiente de color — `_heat_to_bgr(heat)`:**

```python
# OpenCV usa BGR (Blue, Green, Red), no RGB
# heat=0.0: azul  (moving)  -> BGR = (160, 60, 10)
# heat=0.5: verde (pausa)   -> BGR = (0, 200, 50)
# heat=1.0: rojo  (parado)  -> BGR = (0, 0, 210)
```

La interpolacion es lineal en dos tramos:
- `heat < 0.5`: va de azul a verde (el raton empieza a quedarse quieto).
- `heat >= 0.5`: va de verde a rojo (el raton lleva tiempo sin moverse).

**Por que no usar un mapa de calor clastico de densidad (heatmap 2D):**
el enfoque clasico acumula presencia en una imagen de densidad y luego aplica
un blur gaussiano (Gaussian blur). Ese metodo da mapas de calor mas suaves pero
pierde el orden temporal — no se distingue si el raton estuvo quieto al principio
o al final. Este enfoque mantiene la informacion temporal porque el color va
cambiando a lo largo de la trayectoria.

---

### Bloque H — De `.ui` a Python: cómo PyQt5 conecta diseño y lógica

---

#### H.1 El archivo `.ui` y `uic.loadUi()`

**Concepto:** Qt Designer genera un XML (`.ui`) que describe widgets, layouts y propiedades visuales. En tiempo de ejecución, `uic.loadUi()` parsea ese XML y crea en memoria todos los widgets como atributos del objeto Python. A partir de ese momento, `self.btn_video` es exactamente el mismo `QPushButton` que aparece en la ventana — no hay copia, es la misma instancia.

**En el código — `gui/controllers/main_window.py`:**

```python
from PyQt5 import uic

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Lee el XML y crea todos los widgets como atributos de self
        uic.loadUi("gui/main_window.ui", self)

        # Ahora self.btn_video, self.stackedWidget, self.lbl_model_status
        # existen y son los widgets reales de la ventana
        self.btn_video.clicked.connect(self._on_select_video)
```

El nombre del atributo Python es exactamente el `name` del widget en el `.ui`:
```xml
<!-- main_window.ui -->
<widget class="QPushButton" name="btn_video">
```
→ se convierte en `self.btn_video` en Python después de `uic.loadUi()`.

Si cambias el nombre en Qt Designer, tienes que cambiar también todas las referencias en el controlador — es el único punto de acoplamiento entre diseño y lógica.

---

#### H.2 Señales y slots: el patrón de comunicación de Qt

**Concepto:** en PyQt5, los widgets emiten **señales** cuando algo ocurre (clic, cambio de valor, tecla). Tú conectas esa señal a una función Python (el **slot**) usando `.connect()`. Esta conexión es el único mecanismo por el que la UI comunica eventos al controlador. No hay polling, no hay variables globales de estado de botón — todo es reactivo.

**En el código — `gui/controllers/main_window.py`:**

```python
def _connect_signals(self):
    # Widget emite señal → función Python recibe la notificación
    self.btn_video.clicked.connect(self._on_select_video)     # QPushButton.clicked
    self.btn_camera.clicked.connect(self._on_select_camera)   # QPushButton.clicked
    self.btn_lang.clicked.connect(self._toggle_language)      # QPushButton.clicked
    self.spin_width_cm.valueChanged.connect(self._update_ratio_label)  # QDoubleSpinBox

    # Las señales personalizadas del worker también se conectan aquí
    self._worker.frame_ready.connect(self._on_frame_ready)    # pyqtSignal personalizada
    self._worker.finished.connect(self._on_detection_finished)
```

Una señal puede conectarse a varios slots y un slot puede recibir varias señales. La desconexión es automática cuando el objeto se destruye. No hay que "cancelar suscripciones" manualmente.

---

#### H.3 QThread: detección en segundo plano sin congelar la UI

**Concepto:** el event loop de Qt corre en el hilo principal. Si ejecutas una operación lenta (inferencia YOLO frame a frame) en ese hilo, la ventana se congela — no responde a clics, no se repinta. La solución es `QThread`: un hilo separado que hace el trabajo pesado y se comunica con la UI solo mediante señales (que Qt enruta de forma segura entre hilos).

**Regla crítica:** nunca toques un widget directamente desde un QThread. Solo emite señales; el slot en el hilo principal actualiza la UI.

**En el código — `gui/controllers/detect_worker.py`:**

```python
from PyQt5.QtCore import QThread, pyqtSignal

class DetectionWorker(QThread):
    # Señales que el hilo puede emitir (seguras entre hilos)
    frame_ready = pyqtSignal(object, dict, int)   # frame, stats, frame_idx
    log_msg     = pyqtSignal(str)
    finished    = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def run(self):                          # se ejecuta en el hilo secundario
        for res in model.predict(..., stream=True):
            if self._stop:
                break
            # ... procesamiento ...
            self.frame_ready.emit(img, stats, frame_idx)   # notifica a la UI

    def request_stop(self):
        self._stop = True                   # bandera atómica para parar el bucle
```

**En el controlador — `gui/controllers/main_window.py`:**

```python
# Hilo principal: conecta señales y arranca el worker
self._worker = DetectionWorker(source, output_dir, coords_json)
self._worker.frame_ready.connect(self._on_frame_ready)   # slot en hilo principal
self._worker.start()                                      # lanza run() en nuevo hilo

@pyqtSlot(object, dict, int)
def _on_frame_ready(self, frame, stats, frame_idx):
    # Este método corre en el hilo principal → puede tocar widgets con seguridad
    self.lbl_video_feed.setPixmap(...)
    self.lbl_beh_idle.setText(...)
```

El `@pyqtSlot` no es obligatorio pero ayuda a Qt a optimizar el enrutado de señales entre hilos.

---

#### H.4 Convertir frames de OpenCV a QPixmap

**Concepto:** OpenCV trabaja con arrays NumPy en formato `(height, width, 3)` con canales en orden **BGR**. Qt trabaja con `QPixmap` y `QImage` en formato **RGB**. La conversión tiene tres pasos: cambiar orden de canales, crear un `QImage` apuntando a los bytes del array, y envolver en `QPixmap`.

**En el código — `gui/controllers/main_window.py`:**

```python
def _on_frame_ready(self, frame: np.ndarray, stats, frame_idx):
    h, w = frame.shape[:2]

    # 1. Escalar al tamaño del QLabel manteniendo aspecto
    label = self.lbl_video_feed
    scale = min(label.width() / w, label.height() / h)
    dw, dh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (dw, dh))

    # 2. BGR → RGB (OpenCV y Qt discrepan en el orden de canales)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 3. NumPy array → QImage → QPixmap
    #    bytesPerLine = dw * 3  (3 canales, 1 byte cada uno)
    qimg   = QImage(rgb.data, dw, dh, dw * 3, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    label.setPixmap(pixmap)
```

El `QImage` apunta directamente a los bytes del array NumPy sin copiarlos. Por eso `rgb` no puede salir de ámbito mientras `QImage` la está usando — de ahí que se cree `pixmap` en la misma llamada y `rgb` siga viva hasta el final del bloque.

---

#### H.5 Capturar clics del ratón en un QLabel

**Concepto:** `QLabel` no tiene señal `clicked` ni `mousePressEvent` accesible como slot. La forma más directa de interceptar clics en un label desde el controlador es **monkey-patching**: reemplazar el método de la instancia con una función propia. Esto funciona porque Python resuelve métodos de instancia antes que los de clase.

**En el código — `gui/controllers/main_window.py`:**

```python
# En _connect_signals(): sustituye el método de esta instancia concreta
self.lbl_frame_display.mousePressEvent = self._on_calib_click

def _on_calib_click(self, event):
    # event.x(), event.y() son coordenadas dentro del QLabel (píxeles de pantalla)
    # Hay que mapearlas a coordenadas del frame original:
    lx = event.x() - self._calib_offset_x   # descontar margen de letterbox
    ly = event.y() - self._calib_offset_y
    orig_x = int(lx * self._calib_scale_x)  # factor display → imagen original
    orig_y = int(ly * self._calib_scale_y)
    self._calib_points.append((orig_x, orig_y))
```

El mapping es necesario porque el frame se muestra escalado con `Qt.KeepAspectRatio` — el QLabel tiene un tamaño fijo pero la imagen puede dejar barras negras arriba/abajo o izquierda/derecha (`_calib_offset_x/y`), y la escala no es 1:1 (`_calib_scale_x/y`).

---

#### H.6 Navegación entre páginas con QStackedWidget

**Concepto:** `QStackedWidget` es un contenedor que muestra solo una de sus páginas (widgets hijo) a la vez. Es el equivalente a un sistema de pestañas sin las pestañas visibles — tú controlas cuál se muestra desde Python con `setCurrentIndex(n)`. El sidebar actúa como menú de navegación.

**En el código — `gui/controllers/main_window.py`:**

```python
# Índices fijos (definidos por el orden en el .ui)
# 0 = page_home, 1 = page_calibration, 2 = page_detection, 3 = page_results

# Sidebar: cualquier clic navega directamente
self.nav_home.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))

# Flujo guiado: el código navega programáticamente al completar cada paso
def _on_select_video(self):
    # ... carga el frame ...
    self.stackedWidget.setCurrentIndex(1)   # va a calibración

def _on_confirm_calib(self):
    # ... escribe coords.json ...
    self._start_detection()
    self.stackedWidget.setCurrentIndex(2)   # va a detección

def _on_detection_finished(self, output_paths):
    self._show_results(output_paths)
    self.stackedWidget.setCurrentIndex(3)   # va a resultados
```

`nav_detection` y `nav_results` empiezan deshabilitados (`enabled=false` en el `.ui`) y se activan solo cuando hay datos para mostrar — así el usuario no puede saltar a páginas vacías.

---

#### H.7 Estructura de la carpeta `gui/`

```
gui/
├── app.py                      # Entry point: QApplication + MainWindow.show()
│                               # Añade scripts/ al sys.path antes de cualquier import
│
├── main_window.ui              # Diseño Qt Designer (XML)
│                               # Contiene: sidebar, 4 páginas, stylesheet embebido
│
└── controllers/
    ├── __init__.py
    │
    ├── main_window.py          # MainWindow(QMainWindow)
    │   ├── uic.loadUi()        #   — crea todos los widgets como atributos
    │   ├── _connect_signals()  #   — conecta señales a slots
    │   ├── Calibración         #   — captura 4 clics, escribe coords.json
    │   ├── Detección           #   — crea y arranca DetectionWorker
    │   ├── Resultados          #   — carga imágenes, habilita botones Abrir
    │   └── Idioma              #   — aplica TRANSLATIONS[] a todos los widgets
    │
    └── detect_worker.py        # DetectionWorker(QThread)
        ├── pyqtSignal          #   — frame_ready, log_msg, finished, error
        ├── run()               #   — bucle YOLO stream en hilo secundario
        │                       #     replica el núcleo de RatDetector.run()
        │                       #     pero emitiendo señales en cada frame
        └── request_stop()      #   — bandera _stop para salir del bucle limpiamente
```

**Por qué `main_model.py` no desaparece:** es el modo administrador/test. Permite calibrar con el `ImageCalibrator` completo (exterior + interior + 4 agujeros), lanzar entrenamiento, y depurar el pipeline sin interfaz gráfica. La GUI es el modo usuario final — simplificado e independiente.

---

## 1. Orden de lectura de archivos

Lee los archivos en este orden. El orden refleja el flujo real de datos del pipeline, no el alfabético.

| # | Archivo | Qué hace | Recibe de | Pasa a |
|---|---|---|---|---|
| 1 | `scripts/config/interfaces.py` | Define `BaseModule`, la clase abstracta que obliga a cualquier módulo principal a implementar `run()`. Es la única regla de diseño compartida por todos. | — | Todo módulo que herede de él |
| 2 | `scripts/config/config.py` | Centraliza todas las rutas del proyecto (`Paths`) y los parámetros de entrenamiento y detección (`TrainParams`, `DetectParams`) como dataclasses. Es el único sitio donde se cambia una ruta. | `interfaces.py` (importa `BaseModule`) | Todos los demás módulos |
| 3 | `scripts/calibration/calibrator.py` | `ZoneCalibrator` — ventana OpenCV interactiva sobre el primer frame del vídeo. El usuario hace clic para marcar el borde exterior, el interior y los 4 agujeros. Guarda `coords.json`. | Ruta del vídeo (`paths.video_source`) | `datasets/coords.json` |
| 4 | `scripts/calibration/calibrator_image.py` | `ImageCalibrator` — igual que el anterior pero sobre una imagen estática. Añade conversión entre coordenadas de pantalla y coordenadas de imagen original. | Ruta de imagen | `datasets/coords.json` |
| 5 | `scripts/spatial/spatial.py` | `SpatialAnalyzer` — carga `coords.json` y expone métodos geométricos: ¿el snout está dentro del agujero? ¿El bbox sale de la zona interior? ¿El snout está cerca de la pared? | `datasets/coords.json` | `behavior/behavior_classifier.py` |
| 6 | `scripts/detection/trainer.py` | `YOLOTrainer` — lanza el entrenamiento de YOLOv8-Pose con augmentation geométrico puro (sin colores). Copia `best.pt` a `models/yolo_ratas.pt` al terminar. | `config.py` (rutas + `TrainParams`) → `datasets/data.yaml` | `models/yolo_ratas.pt` |
| 7 | `scripts/behavior/behavior_classifier.py` | `_LabelStabilizer` + `BehaviorClassifier` — aplica la cadena de reglas (head_dipping → horizontal → climbing → rearing) y filtra el parpadeo de etiquetas por histéresis temporal. | `yolo_label`, `speed`, `snout_kp`, `rat_box`, `SpatialAnalyzer` | `final_label` (string) → `detector.py` |
| 8 | `scripts/output/writers.py` | `VideoOutput` + `CsvOutput` — encapsulan la apertura, escritura y cierre del MP4 y el CSV. No calculan nada, solo persisten. | Datos ya calculados por `detector.py` | `outputs/detections/…/*.mp4` + `*.csv` |
| 9 | `scripts/detection/detector.py` | `_SpeedTracker` + `RatDetector` — orquestador principal. Carga el modelo YOLO, recorre el vídeo frame a frame, extrae bbox y keypoints, calcula velocidad, delega la clasificación en `BehaviorClassifier`, delega el dibujo, y delega la escritura en `writers.py`. | `models/yolo_ratas.pt`, `SpatialAnalyzer`, `BehaviorClassifier`, `VideoOutput`, `CsvOutput` | `outputs/…/*.mp4` + `*.csv` |
| 10 | `scripts/utils/stats_generator.py` | `StatsGenerator` — lee el CSV de salida y genera: imagen de trayectoria, mapa de calor dinamico (color por tiempo parado) y Excel con metricas OFT (bouts, duraciones, habituacion, zonificacion). | `outputs/.../*.csv` + `coords.json` | `stats/trajectory_*.png` + `stats/heatmap_*.png` + `stats/stats_*.xlsx` |
| 11 | `scripts/main_model.py` | Menu interactivo (modo admin/test). Opciones: 1=Entrenar, 2=Detectar video, 3=Detectar camara en vivo, 4=Salir. Logica de calibracion compartida via `_run_detection()`. | Todo lo anterior | Todo lo anterior |
| 12 | `gui/main_window.ui` | Diseño Qt Designer (XML). Define sidebar, 4 páginas (Home, Calibración, Detección, Resultados), stylesheet embebido y nombres de todos los widgets. Es solo estructura visual — no tiene lógica. | — | `gui/controllers/main_window.py` vía `uic.loadUi()` |
| 13 | `gui/controllers/detect_worker.py` | `DetectionWorker(QThread)` — replica el núcleo del bucle de `RatDetector.run()` en un hilo secundario. Emite señales PyQt (`frame_ready`, `finished`, `error`) en lugar de escribir en consola. Llama a `StatsGenerator` al terminar. | `source` (Path o int cámara), `output_dir`, `coords_json` | Señales → `main_window.py`; archivos en `outputs/detections/<run>/` |
| 14 | `gui/controllers/main_window.py` | `MainWindow(QMainWindow)` — carga el `.ui`, conecta todas las señales, orquesta la navegación entre páginas, gestiona la calibración con clics sobre el frame, convierte frames OpenCV→QPixmap y soporta cambio de idioma ES/EN. | `main_window.ui` + `DetectionWorker` | La ventana visible al usuario final |
| 15 | `gui/app.py` | Entry point de la GUI. Añade `scripts/` al `sys.path`, crea `QApplication` y muestra `MainWindow`. Se lanza con `python -m gui.app` desde la raíz del proyecto. | — | Ventana gráfica |
| — | `adminScripts/DECISIONES_TFG.md` | **Referencia.** Decisiones de diseño del TFG — justificaciones de cada elección arquitectónica. Solo lectura, no es código. | — | — |

---

## 2. Diagrama de flujo en texto

Quién llama a quién y qué formato de datos viaja entre módulos.

```
Usuario
  └─► main_model.py  (menú interactivo)
        │
        ├─ Opción 1 — Calibrar
        │    └─► calibration/calibrator.py   (vídeo)
        │         o calibration/calibrator_image.py  (imagen)
        │              │
        │              └─► datasets/coords.json
        │                   {exterior, interior, holes, hole_radius,
        │                    limits_inner, limits_outer}
        │
        ├─ Opción 2 — Entrenar
        │    └─► detection/trainer.py
        │              │  (lee datasets/data.yaml, escribe en runs/train/)
        │              └─► models/yolo_ratas.pt
        │
        ├─ Opción 2 — Detectar (video)
        │    │
        │    ├─► calibration/calibrator_image.py  (primer frame del video)
        │    │         └─► datasets/coords.json
        │    │
        │    └─► detection/detector.py  (RatDetector, camera_index=None)
        │
        └─ Opción 3 — Detectar (camara en vivo)
             │
             ├─► cv2.VideoCapture(0) — un frame para calibracion
             ├─► calibration/calibrator_image.py
             │         └─► datasets/coords.json
             │
             └─► detection/detector.py  (RatDetector, camera_index=0)
                  source_yolo=0 -> YOLO abre la camara directamente
                  │
                  ├── carga spatial/spatial.py  ← datasets/coords.json
                  │         SpatialAnalyzer: check_dipping(), check_sniffing_wall(),
                  │                          bbox_entirely_inside(), is_valid_for()
                  │
                  ├── carga behavior/behavior_classifier.py
                  │         BehaviorClassifier(spatial_logic=SpatialAnalyzer)
                  │         _LabelStabilizer (estado interno, frame a frame)
                  │
                  ├── crea output/writers.py
                  │         VideoOutput  → MP4 principal + MP4 limpio (dual_output)
                  │         CsvOutput   → CSV con cabecera HEADER
                  │
                  └── bucle por frame (YOLO stream):
                       │
                       ├── YOLO Pose → res.boxes + res.keypoints
                       │         yolo_label: str  (ej. "rat_horizontal")
                       │         rat_box:    ndarray [x1,y1,x2,y2]
                       │         snout_kp:   ndarray [x,y]  o  None
                       │         tail_kp:    ndarray [x,y]  o  None
                       │
                       ├── _SpeedTracker.update(rat_box, w, h)
                       │         → speed: float  (desplazamiento normalizado × 100)
                       │
                       ├── BehaviorClassifier.classify(yolo_label, speed,
                       │                               snout_kp, rat_box, spatial_ok)
                       │         → final_label: str  (ej. "sniffing_walking")
                       │
                       ├── _get_color(final_label) + cv2.rectangle/putText
                       │
                       ├── VideoOutput.write(img, img_clean)
                       │         → frame escrito en MP4
                       │
                       └── CsvOutput.write_row(frame_idx, fps, …)
                                 → fila escrita en CSV

             Tras el bucle:
             └─► utils/stats_generator.py  ← CSV generado + coords.json
                       ├── stats/trajectory_{stem}.png
                       └── stats/stats_{stem}.xlsx


── MODO GUI (gui/app.py  →  gui/controllers/main_window.py) ─────────

Usuario (farmacóloga)
  └─► gui/app.py           QApplication + MainWindow.show()

       MainWindow           carga main_window.ui con uic.loadUi()
         │
         ├─ page_home
         │    ├── btn_video  → QFileDialog  →  _video_source = Path(archivo)
         │    └── btn_camera →  cv2.VideoCapture(0).read()  →  _video_source = 0
         │
         ├─ page_calibration
         │    ├── lbl_frame_display  ← primer frame (video) o snapshot (cámara)
         │    │   mousePressEvent → 4 clics → _calib_points [(x,y), ...]
         │    ├── spin_width_cm / spin_height_cm → px/cm ratio en pantalla
         │    └── btn_confirm_calib
         │          ├── escribe datasets/coords.json  (exterior + dimensiones)
         │          ├── crea  outputs/detections/<stem>_<timestamp>/
         │          └── lanza DetectionWorker (QThread)
         │
         ├─ page_detection
         │    │
         │    │   DetectionWorker.run()  [hilo secundario]
         │    │     ├── YOLO.predict(stream=True)
         │    │     ├── BehaviorClassifier + _SpeedTracker
         │    │     ├── VideoOutput → *_anotado.mp4 + *_limpio.mp4
         │    │     ├── CsvOutput   → *.csv
         │    │     ├── emit frame_ready(img, stats, idx)  → _on_frame_ready()
         │    │     │        lbl_video_feed.setPixmap(cv2→QPixmap)
         │    │     │        lbl_beh_* .setText(segundos acumulados)
         │    │     └── emit finished(output_paths)  → _on_detection_finished()
         │    │
         │    ├── lbl_video_feed   (frame en vivo, BGR→RGB→QPixmap)
         │    ├── lbl_elapsed_time (QTimer cada 1 s)
         │    └── btn_stop → worker.request_stop()  (_stop = True)
         │
         └─ page_results
              ├── tab_images
              │    ├── lbl_trajectory_img ← stats/trajectory_*.png
              │    └── lbl_heatmap_img    ← stats/heatmap_*.png
              ├── grp_files → btn_open_* → os.startfile(path)
              │    ├── Estadísticas .xlsx
              │    ├── Vídeo anotado
              │    ├── Vídeo de recorrido
              │    └── Carpeta de salida
              └── btn_new_detection → vuelve a page_home, resetea estado
```

---

## 3. Ejercicios prácticos

Cada ejercicio se puede hacer en 30-45 minutos sin GPU ni dataset completo.

---

### Grupo 1 — `config/interfaces.py` + `config/config.py`

**Objetivo:** entender cómo un dataclass actúa como sistema de configuración centralizado y por qué es mejor que variables globales sueltas.

**Ejercicio:**

```python
import sys
sys.path.insert(0, "ruta/al/proyecto/scripts")

from config.config import paths
import dataclasses

# Imprime todas las rutas configuradas
for campo, valor in dataclasses.asdict(paths).items():
    print(f"  {campo:<20} {valor}")
```

Ejecuta esto y responde en papel:
1. ¿Cuántas rutas distintas gestiona `paths`?
2. Cambia manualmente `paths.video_source` a una ruta inventada y comprueba que `paths.output_video` no cambia — ¿qué implica eso sobre cómo se calcula `output_video`?
3. ¿Qué ocurre si llamas a `paths.check_dirs()` cuando la carpeta `models/` no existe?

---

### Grupo 2 — `calibration/calibrator.py` + `calibration/calibrator_image.py`

**Objetivo:** entender qué produce la calibración y qué espera recibir `SpatialAnalyzer`.

**Ejercicio:**

Crea un archivo `coords_test.json` a mano con esta estructura (usa valores inventados):

```json
{
  "exterior": [[50, 50], [750, 650]],
  "interior": [[100, 100], [700, 600]],
  "holes": [[200, 200], [500, 200], [200, 450], [500, 450]],
  "hole_radius": 25,
  "limits_inner": {"x_min": 100, "x_max": 700, "y_min": 100, "y_max": 600},
  "limits_outer": {"x_min": 50,  "x_max": 750, "y_min": 50,  "y_max": 650}
}
```

Luego carga el JSON y responde:
1. ¿Qué representan físicamente `exterior` e `interior` en la caja real?
2. ¿Por qué hay exactamente 4 agujeros?
3. ¿Qué diferencia hay entre `exterior` (lista de 2 puntos) y `limits_outer` (dict con x_min/x_max/y_min/y_max)? ¿Cuál es más cómodo para buscar si un punto está dentro?

---

### Grupo 3 — `spatial/spatial.py`

**Objetivo:** entender que `head_dipping` es distancia euclidiana, no magia de red neuronal.

**Ejercicio:**

Dado un agujero en `[320, 240]` con radio 25 px y `DIPPING_RATIO = 0.9`, el umbral real es `25 × 0.9 = 22.5 px`. Calcula a mano si cada uno de estos puntos activa `check_dipping()`:

| Punto | Distancia al centro | ¿Dipping? |
|---|---|---|
| `[320, 241]` | ? | ? |
| `[335, 255]` | ? | ? |
| `[344, 240]` | ? | ? |
| `[400, 300]` | ? | ? |

Fórmula: `d = sqrt((x - 320)² + (y - 240)²)`. El punto activa dipping si `d < 22.5`.

Luego verifica cargando el JSON de test del ejercicio anterior:

```python
from pathlib import Path
from spatial.spatial import SpatialAnalyzer
import sys; sys.path.insert(0, "ruta/al/proyecto/scripts")

sa = SpatialAnalyzer(config_path=Path("coords_test.json"))
for pt in [[320,241], [335,255], [344,240], [400,300]]:
    print(pt, sa.check_dipping(pt))
```

---

### Grupo 4 — `detection/trainer.py`

**Objetivo:** entender que el augmentation no es arbitrario — refleja restricciones físicas del entorno.

**Ejercicio:** Lee el bloque de parámetros de `self.model.train()` en `trainer.py` y responde en papel (no necesitas ejecutar nada):

1. `degrees=180` — ¿por qué 180 y no 90? ¿Qué postura de rata sería inválida con 90?
2. `hsv_h=0.0, hsv_s=0.0, hsv_v=0.0` — ¿por qué se desactiva el augmentation de color? ¿Qué pasaría si la iluminación fuese variable?
3. `mosaic=0.0` — el mosaic de YOLO mezcla 4 imágenes en una. ¿Por qué es problemático en este dataset?
4. `erasing=0.0` — el erasing borra regiones aleatorias. ¿Qué parte de la rata podría borrarse que no queremos perder?

Bonus: busca en la documentación de Ultralytics qué hace `cos_lr=True` y en qué situaciones ayuda.

---

### Grupo 5 — `detection/detector.py`

**Objetivo:** entender el flujo frame a frame de un detector y qué significa "orquestar sin implementar".

**Ejercicio:** Lee `RatDetector.run()` y crea en papel este diagrama para un frame en el que YOLO detecta una rata con `yolo_label="rat_horizontal"` y `speed=0.05`:

```
Frame N
  ├── YOLO detecta → rat_box = [x1,y1,x2,y2],  yolo_label = "rat_horizontal"
  ├── keypoints    → snout_kp = [sx, sy],        tail_kp = [tx, ty]
  ├── _SpeedTracker → speed = 0.05
  ├── BehaviorClassifier.classify(…) → final_label = ???
  ├── _get_color(final_label) → color = ???
  ├── VideoOutput.write(img, img_clean)
  └── CsvOutput.write_row(…)
```

Rellena los `???` a mano usando los umbrales de `behavior_classifier.py` (STILL_SPEED_THRESHOLD = 0.15).

Luego responde: si YOLO no detecta ninguna rata en un frame concreto (rat_box es None), ¿qué escribe ese frame en el CSV?

---

### Grupo 6 — `behavior/behavior_classifier.py`

**Objetivo:** entender los umbrales, el aspect ratio de rearing, y la histéresis del estabilizador.

**Ejercicio:**

```python
import sys; sys.path.insert(0, "ruta/al/proyecto/scripts")
import numpy as np
from behavior.behavior_classifier import BehaviorClassifier

clf = BehaviorClassifier(spatial_logic=None)  # sin análisis espacial

casos = [
    # (yolo_label, speed, snout_kp, rat_box, spatial_ok)
    ("rat_horizontal", 0.50, None, np.array([0,0,100,50]),  False),
    ("rat_horizontal", 0.10, None, np.array([0,0,100,50]),  False),
    ("rat_rearing",    0.20, None, np.array([100,50,140,130]), False),  # h=80, w=40 -> ratio=2.0
    ("rat_rearing",    0.20, None, np.array([100,50,200,130]), False),  # h=80, w=100 -> ratio=0.8
    ("rat_rearing",    0.20, None, np.array([100,50,200,160]), False),  # h=110, w=100 -> ratio=1.1
]

# Predice el resultado ANTES de ejecutar, luego comprueba
for yolo_label, speed, snout_kp, rat_box, spatial_ok in casos:
    result = clf.classify(yolo_label, speed, snout_kp, rat_box, spatial_ok)
    print(f"  {yolo_label:20} speed={speed:.2f}  bbox_h/w={round((rat_box[3]-rat_box[1])/(rat_box[2]-rat_box[0]),2):<5}  → {result}")
```

Luego: crea un bucle que llame 10 veces seguidas con `yolo_label="rat_grooming"` y imprime la etiqueta estable en cada iteración. ¿En qué frame cambia la etiqueta por primera vez? Compara con el parámetro `fast_labels={"rat_grooming": 3}` del constructor.

---

### Grupo 7 — `output/writers.py`

**Objetivo:** entender cómo se desacopla la escritura de datos del cálculo.

**Ejercicio:**

```python
import sys; sys.path.insert(0, "ruta/al/proyecto/scripts")
import numpy as np
from pathlib import Path
from output.writers import CsvOutput

csv_out = CsvOutput(Path("test_output.csv"))

# Escribe 3 filas inventadas
csv_out.write_row(0,  30.0, "rat_horizontal", "walking",
                  box=np.array([100,200,300,350]),
                  snout_kp=np.array([200.0, 210.0]),
                  tail_kp=np.array([250.0, 330.0]),
                  speed=0.42)

csv_out.write_row(1,  30.0, "rat_horizontal", "immobile",
                  box=np.array([102,201,305,352]),
                  snout_kp=np.array([201.0, 212.0]),
                  tail_kp=None,
                  speed=0.08)

csv_out.write_row(30, 30.0, "rat_rearing",    "rat_rearing",
                  box=np.array([150,100,220,280]),
                  snout_kp=None,
                  tail_kp=np.array([185.0, 270.0]),
                  speed=0.0)

csv_out.close()

import pandas as pd
df = pd.read_csv("test_output.csv")
print(df)
print("\nColumnas:", list(df.columns))
```

Responde:
1. ¿Qué valor aparece en `snout_x` cuando `snout_kp=None`? ¿Por qué -1.0 y no 0.0 o NaN?
2. ¿Qué pasa si llamas `write_row()` después de `close()`?

---

### Grupo 8 — `utils/stats_generator.py`

**Objetivo:** entender el post-procesado estadístico y la diferencia entre métricas de comportamiento y métricas de movimiento.

**Ejercicio:** Localiza un CSV real en `outputs/detections/…/` (o usa el CSV que generaste en el ejercicio anterior, aunque sea corto). Luego:

```python
import sys; sys.path.insert(0, "ruta/al/proyecto/scripts")
from pathlib import Path
from utils.stats_generator import StatsGenerator

gen = StatsGenerator(Path("ruta/al/archivo.csv"), coords_json=None)
gen.generate(Path("stats_test/"))
```

Abre la imagen `stats_test/trajectory_*.png`. Responde:
1. ¿Qué columna del CSV usa para dibujar la trayectoria (`tail_x/tail_y` o `snout_x/snout_y`)? Busca en `_generate_trajectory()`.
2. ¿Por qué la cola y no el hocico? Piensa en qué parte del cuerpo se mueve más suavemente durante la locomoción.
3. Cambia `COLOR_TRACK = (0, 200, 0)` a `(0, 0, 255)` en el código y regenera. ¿Qué color sale?

---

## 4. Checklist de autoevaluación

Al final de la semana, sin mirar el código, responde estas preguntas con tus propias palabras. Si puedes explicarlo sin abrir ningún archivo, lo entiendes.

1. ¿Por qué el proyecto usa YOLOv8-**Pose** en vez de YOLOv8 estándar? ¿Qué añade la estimación de pose que no tiene un detector de bounding boxes?

2. ¿Qué contiene `coords.json`? Describe los 6 campos principales y qué representa cada uno en la caja física.

3. ¿Por qué `rat_horizontal` se mapea a tres comportamientos distintos (walking / immobile / sniffing)? ¿Qué información adicional resuelve la ambigüedad?

4. ¿Cómo calcula `_SpeedTracker` la velocidad? ¿Por qué usa la media de los últimos N frames y no la velocidad instantánea?

5. Explica el orden de prioridad en `BehaviorClassifier.classify()`. Si YOLO dice `rat_head_dipping` pero el snout no está sobre ningún agujero, ¿cuál es el resultado?

6. ¿Qué hace `_LabelStabilizer`? Si una etiqueta nueva aparece solo 2 frames seguidos y el umbral es 8, ¿qué etiqueta devuelve en esos 2 frames?

7. ¿Qué problema resuelve `output/writers.py` que antes resolvía `detector.py` inline? ¿Qué principio de diseño representa esa separación?

8. ¿Qué columnas tiene el CSV de salida? Sin mirar el código, nombra al menos 8.

9. ¿Por qué se desactivó el augmentation de color en el entrenamiento de YOLO? ¿Qué consecuencia tendría activarlo?

10. ¿Por qué se eligió Recall sobre Precision como métrica prioritaria? ¿Qué significa un falso negativo en el contexto de un experimento conductual?

11. ¿Cuál fue el problema fundamental de la RNN (no el técnico, sino el conceptual)? Explica por qué la supervisión circular invalida el entrenamiento.

12. ¿Qué representa un "bout" en las métricas OFT del Excel? ¿Cómo se diferencia de la duración total?

13. Dada la trayectoria de ejemplo del README, ¿qué dice del comportamiento del animal la concentración de líneas en los bordes del recuadro?

14. Si alguien te dice "el modelo tiene mAP50 = 0.87, ¿es mejor que uno con 0.82?", ¿qué responderías basándote en la decisión entre exp8 y exp9?

15. ¿Qué pasaría si ejecutases la detección sin calibrar primero (`coords.json` no existe)? Traza el camino en el código desde `detector.py` hasta `SpatialAnalyzer._load_config()`.

---

## Valoración de asumibilidad

**Veredicto honesto: una semana es ajustada pero factible**, con estas condiciones:

**Lo que es cómodo en 7 días:**
- Grupos 1–3 (config, calibración, spatial): conceptualmente simples, solo Python y geometría. 1–2 días.
- Grupos 5–7 (detector, clasificador, writers): el núcleo del pipeline. 2–3 días.
- Checklist de autoevaluación + repaso: 1 día.

**Lo que es arriesgado:**
- **Grupo 4 (trainer):** entender los parámetros de augmentation de YOLO requiere leer documentación externa. Si el tiempo aprieta, sustitúyelo por una lectura comentada de 20 minutos sin hacer el bonus.
- **Grupo 8 (stats_generator):** el archivo es largo (430 líneas). Céntrate solo en `_generate_trajectory()` y `_count_bouts()`. El resto del Excel es repetitivo una vez entiendes la estructura.

**Qué ajustaría si la semana es corta:**
1. En el Grupo 8, haz solo el ejercicio de `_generate_trajectory()` y omite el análisis del Excel.
2. Para el Grupo 4, lee el código pero omite el bonus de `cos_lr`.

**Expectativa real al cabo de una semana:** deberías poder explicar el pipeline principal (grupos 1–7) con tus propias palabras, trazar el camino de datos desde el vídeo hasta el CSV, y responder 10 de las 15 preguntas de la checklist. Stats y RNN son bonus.
