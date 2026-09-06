# Teoría del modelo — Guía de estudio personal

> Para imprimir y leer con calma. Todo está conectado con el código real del proyecto.

---

## 1. CNN — lo que hace YOLO (ver un frame)

Una **CNN (Red Neuronal Convolucional)** procesa una sola imagen.
Aplica filtros que se deslizan sobre la imagen buscando patrones:
- Capas iniciales: bordes y texturas (pelo del ratón, fondo negro)
- Capas finales: formas complejas (silueta del cuerpo, postura)

Al final produce una respuesta **para ese único fotograma**:
"aquí hay una rata, está en posición horizontal, confianza 87%."

**No tiene memoria.** Cada frame es completamente independiente.

**En el proyecto:** YOLO ve un frame → devuelve bbox + 3 keypoints (hocico, lomo, cola) + clase.

---

## 2. RNN / LSTM — procesar secuencias en el tiempo

Una **RNN (Red Neuronal Recurrente)** procesa *secuencias*.
Mantiene un **estado interno** (memoria) que se actualiza con cada nuevo elemento.
La predicción del frame 30 tiene en cuenta lo que pasó en los frames 1 al 29.

Una **LSTM (Long Short-Term Memory)** es un tipo de RNN más sofisticado que resuelve
el problema del "olvido a largo plazo": decide activamente qué recordar y qué olvidar.

**Analogía directa:**
- CNN = una fotografía
- RNN/LSTM = una película

**Por qué parecía útil aquí:**
`rat_horizontal` es ambiguo en una sola imagen — un ratón quieto y uno caminando
se ven idénticos desde arriba. Una LSTM que vea 30 frames seguidos puede detectar
si el centroide se ha desplazado → walking, o si lleva quieto → immobile.

---

## 3. Por qué se descartó la RNN — 4 problemas estructurales

### Problema 1 — Supervisión circular (el más grave)
Para entrenar la LSTM necesitas datos etiquetados.
Esos datos vendrían del CSV que genera el propio detector YOLO.
Si YOLO etiqueta mal el 40% de los frames de grooming (que es lo que pasa),
la LSTM aprende esos errores como si fueran correctos.
No aprende el comportamiento real — aprende los fallos de YOLO.
Esto se llama **label noise propagation** y no tiene solución sin etiquetado manual independiente.

### Problema 2 — Características insuficientes
La LSTM usaba 5 números por frame: cx, cy, w, h, velocidad (todos del bbox).
Pero el bbox es ciego a la postura interna: un rearing y un immobile pueden tener
el mismo bbox si el ratón es pequeño en vertical.
La información discriminativa real son los **keypoints** (hocico, lomo, cola) —
que YOLO-Pose sí extrae, pero que la LSTM ignoraba.

### Problema 3 — Clases incompatibles
La LSTM producía: `rat_horizontal`, `rat_climbing`, `rat_rearing`, `rat_grooming`, `rat_head_dipping`.
El pipeline final necesita: `walking`, `immobile`, `sniffing` —
derivados de `rat_horizontal` mediante velocidad y geometría.
La LSTM no tenía representación de esos estados.

### Problema 4 — Nunca llegó a entrenarse
El fichero `models/best_rnn.pth` nunca existió.
El código detecta su ausencia y activa `self.active = False`.
Contribución real de la RNN en todos los experimentos: **cero.**

---

## 4. Qué la reemplaza: `_LabelStabilizer` (histéresis temporal)

**El problema real no era falta de contexto temporal — era ruido de clasificación.**
YOLO alterna entre `rat_grooming` e `immobile` en frames consecutivos aunque el ratón no se mueva,
porque en una sola imagen ambas posturas se parecen.

Eso no necesita aprendizaje. Necesita un **filtro.**

`_LabelStabilizer` implementa **histéresis temporal:**
una etiqueta nueva solo reemplaza a la actual si aparece durante **8 frames consecutivos** (~0,27 s).
Si aparece menos, el sistema mantiene la etiqueta anterior.

**Analogía del termostato:**
- Sin histéresis: si el objetivo es 20°C, el termostato enciende a 19,9°C y apaga a 20,1°C → oscilación constante.
- Con histéresis: enciende cuando baja de 18°C y apaga cuando sube a 22°C → estabilidad.

**Por qué es mejor que la LSTM en este caso:**
| | LSTM | _LabelStabilizer |
|---|---|---|
| Datos de entrenamiento | Necesita miles de frames etiquetados | Ninguno |
| Supervisión circular | Sí, inevitable con CSVs del detector | No aplica |
| Inercia correcta | Aprendida (puede ser incorrecta) | Fijada a priori (0,27 s por etología) |
| Interpretable | No (caja negra) | Sí (regla explícita) |

**Código:** `scripts/behavior/behavior_classifier.py` → clase `_LabelStabilizer`

**Cómo explicárselo al tutor (3-4 frases):**
> "Implementé YOLO-Pose para extraer los keypoints del esqueleto, y empecé a diseñar
> una LSTM para el contexto temporal como habíamos comentado. Al profundizar encontré
> un problema estructural: el único origen de datos para entrenarla eran los CSVs del
> propio detector, lo que crea un ciclo vicioso donde la red aprende los errores de YOLO
> en lugar del comportamiento real. Lo resolví con un filtro de histéresis determinista:
> una etiqueta solo se acepta si aparece 8 frames seguidos, lo que elimina el parpadeo
> sin necesitar entrenamiento. El resultado es más robusto y explicable que una LSTM
> entrenada con datos ruidosos."

---

## 5. YOLO-Pose — cómo funciona la teoría base

**YOLO (You Only Look Once):**
Divide la imagen en una cuadrícula. Para cada celda predice:
- ¿Hay un objeto aquí? (objectness)
- ¿Dónde exactamente? (bounding box: x, y, w, h)
- ¿Qué clase es? (clasificación)
Todo en un solo paso por la red ("Only Once"), de ahí el nombre.

**YOLOv8-Pose añade una rama extra:**
Además del bbox y la clase, predice la posición de **keypoints** (puntos del esqueleto).
Cada keypoint tiene: coordenadas (x, y) + confianza de visibilidad (0–1).

**En el proyecto:**
- Keypoint 0 = snout (hocico) — el más importante: decide head_dipping y sniffing
- Keypoint 1 = spine (columna / lomo)
- Keypoint 2 = tail (cola) — se usa para la trayectoria en el Excel

**Transfer learning:**
YOLO no se entrena desde cero. Parte de pesos preentrenados en COCO
(80 clases, millones de imágenes). El fine-tuning solo ajusta las capas finales
para las 5 clases de postura del ratón. Con 248 imágenes originales esto es posible;
sin transfer learning, sería imposible.

---

## 6. Métricas al entrenar con YOLO-Pose

Todas visibles en `runs/pose/train/expN/results.csv` y en las gráficas de Ultralytics.

### Métricas de error (Loss) — bajan durante el entrenamiento, mejor cuanto más bajas

| Métrica | Qué mide | Target |
|---|---|---|
| `box_loss` | Error en las coordenadas del bounding box | < 1,5 |
| `pose_loss` | Distancia entre keypoints predichos y reales | < 1,5 |
| `kobj_loss` | Si el modelo predice correctamente si un keypoint es visible | < 1,0 |
| `cls_loss` | Error al clasificar la postura (grooming vs rearing, etc.) | < 1,0 |
| `dfl_loss` | Refinamiento a nivel de píxel del bbox | < 1,5 |

### Métricas de validación — suben durante el entrenamiento, mejor cuanto más altas (0–1)

| Métrica | Qué mide | Target |
|---|---|---|
| `Precision (P)` | De todas las detecciones que hizo, ¿cuántas fueron correctas? (evita falsas alarmas) | > 0,80 |
| `Recall (R)` | De todos los ratones reales en el vídeo, ¿cuántos detectó? (evita omisiones) | > 0,80 |
| `mAP50` | Precisión media con solapamiento ≥ 50%. Estándar del sector. | > 0,85 |
| `mAP50-95` | Más exigente (exige solapamiento del 50% al 95%). 0,5–0,7 = éxito en pose. | > 0,50 |

**Por qué Recall > Precision en este proyecto:**
Un frame no detectado es un dato perdido para siempre en el análisis conductual.
Un falso positivo ocasional lo filtra la lógica espacial después.
Por eso se bajó `conf_threshold` a 0,18 — para maximizar recall aunque acepte algo más de ruido.

**Resultado alcanzado (exp8, modelo activo):** mAP50 = 0,82 | Recall = 0,887 | Precision = 0,688

---

## 7. Archivos del proyecto donde ver métricas, Excel y canvas

### Métricas de entrenamiento
- **Archivo CSV bruto:** `runs/pose/train/expN/results.csv`
  Ábrelo en Excel, grafica `metrics/mAP50(B)` por epoch — verás el pico donde se guardó `best.pt`.
- **Comparativa entre experimentos:** `adminScripts/_run_validate.py`
  Ejecuta detección sobre dos vídeos y muestra la distribución de comportamientos lado a lado.

### Excel de métricas OFT y canvas de trayectoria
- **Archivo principal:** `scripts/utils/stats_generator.py`
  - Método `_generate_trajectory()` — dibuja el recorrido del ratón (cola frame a frame)
    sobre un canvas en blanco del tamaño de la caja. La cola es el punto más estable
    del cuerpo durante la locomoción, por eso se usa y no el hocico.
  - Método `_generate_excel()` — genera el Excel con bouts por comportamiento,
    duración total en segundos, índices OFT (thigmotaxis, habituation, etc.).
  - Método `_count_bouts()` — agrupa frames consecutivos del mismo comportamiento
    en episodios. Un "bout" = episodio continuo de al menos N frames del mismo estado.

### Lógica espacial (agujeros, paredes)
- `scripts/spatial/spatial.py` → `check_dipping()` y `check_sniffing_wall()`
  Geometría pura: distancia euclidiana del hocico al centro del agujero vs radio.

---

## 8. Recursos visuales recomendados

### Canales de YouTube

| Canal | Idioma | Para qué |
|---|---|---|
| **Dot CSV** | Español | Intuición de ML y redes neuronales desde cero. Empezar aquí. |
| **3Blue1Brown** | Inglés (hay subtítulos) | Serie "Neural Networks" — la mejor visualización matemática de cómo aprende una red |
| **StatQuest with Josh Starmer** | Inglés | Vídeo específico de LSTMs. Muy claro y paso a paso. |

**Búsquedas concretas en YouTube:**
- `"3blue1brown neural network"` — serie de 4 vídeos, ver del 1 al 4
- `"statquest LSTM"` — un solo vídeo, cubre todo lo necesario
- `"dot csv que es una red neuronal"` — punto de entrada en español
- `"yolov8 pose estimation tutorial"` — para ver YOLO-Pose en acción

### Documentación oficial
- **Ultralytics (YOLO):** https://docs.ultralytics.com
- **YOLO-Pose específico:** https://docs.ultralytics.com/tasks/pose/
- **Métricas de entrenamiento:** https://docs.ultralytics.com/guides/yolo-performance-metrics/
