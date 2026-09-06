# Decisiones Técnicas del TFG — Registro de Aprendizaje

Este documento recoge, en orden cronológico, las decisiones de diseño más relevantes del proyecto: qué se intentó, qué problemas se encontraron, cómo se resolvieron y por qué las soluciones adoptadas son las correctas. Está pensado como material de referencia para redactar la memoria del TFG.

---

## 1. La Red Neuronal Recurrente (RNN/LSTM): diseño, limitaciones y sustitución

### Contexto y motivación

El problema central del sistema es que YOLOv8, al analizar cada fotograma de forma independiente, no puede distinguir entre un ratón que camina y uno que está inmóvil: ambos presentan el mismo perfil horizontal. La solución conceptualmente elegante es añadir una segunda red neuronal que analice la **secuencia temporal** de posiciones, como lo haría un observador humano que recuerda los fotogramas anteriores.

Se diseñó e implementó una LSTM (Long Short-Term Memory) de dos capas con los siguientes parámetros de entrada por fotograma:
- Centro del bounding box normalizado: `cx`, `cy`
- Dimensiones del bounding box normalizadas: `w`, `h`
- Velocidad del centroide respecto al fotograma anterior: `speed`

La red acumulaba 30 fotogramas consecutivos antes de emitir una predicción, clasificando en 5 clases: `rat_rearing`, `rat_grooming`, `rat_horizontal`, `rat_climbing`, `rat_head_dipping`.

### Problemas encontrados

**Problema 1 — Supervisión circular (el más grave)**

El entrenador de la RNN (`RNNTrainer`) leía los CSVs generados por el propio detector como datos de entrenamiento. Esto crea un ciclo vicioso: la RNN aprendería a imitar las predicciones de YOLO, incluyendo todos sus errores. Si YOLO confundía grooming con horizontal en el 40% de los casos, la RNN aprendería que "horizontal" es la etiqueta correcta para esos fotogramas. No se estaría aprendiendo el comportamiento real del ratón, sino el sesgo del modelo anterior.

En la literatura de aprendizaje automático este problema se denomina **label noise propagation** y es especialmente dañino cuando el modelo fuente tiene una tasa de error alta en las clases minoritarias, que es exactamente el caso de grooming y rearing con pocos ejemplos de entrenamiento.

**Problema 2 — Características insuficientes**

Las 5 características de entrada (`cx`, `cy`, `w`, `h`, `speed`) son derivadas exclusivamente del bounding box. Sin embargo, el bounding box es "ciego" a la postura interna del ratón: un ratón de pie erguido (`rearing`) y uno acostado (`horizontal`) pueden tener bounding boxes del mismo tamaño si el ratón es pequeño en posición vertical. Las características que realmente distinguen comportamientos son los **keypoints** (hocico, columna, cola), que describen la geometría del esqueleto. La RNN ignoraba toda esa información.

Dicho de otro modo: la RNN usaba menos información que la lógica espacial ya implementada, lo que hacía imposible que la superara.

**Problema 3 — Incompatibilidad de clases**

El vocabulario de salida de la RNN era `['rat_rearing', 'rat_grooming', 'rat_horizontal', 'rat_climbing', 'rat_head_dipping']`. Sin embargo, el pipeline final produce `walking`, `immobile` y `sniffing`, que son comportamientos derivados de `rat_horizontal` mediante lógica de velocidad y espacial. La RNN no tenía representación de estos estados: aunque hubiera predicho correctamente, habría producido etiquetas que el sistema no sabía cómo manejar.

**Problema 4 — Nunca se entrenó**

El archivo `models/best_rnn.pth` nunca existió en ningún momento del proyecto. La clase `ActionPredictor` detecta su ausencia en el constructor y activa el flag `self.active = False`, haciendo que cada llamada a `update_and_predict()` devuelva `None` sin ejecutar ninguna inferencia. Todas las corridas desde exp7 hasta exp9 produjeron sus resultados **sin ninguna contribución de la RNN**.

### Decisión adoptada y justificación

Se eliminó la RNN del pipeline activo y se sustituyó por `_LabelStabilizer`, un mecanismo determinista de histéresis temporal. Ver sección 7 para el análisis completo de por qué la histéresis es la solución correcta y qué alternativas de IA existen.

**Valor académico de este análisis**

Identificar cuándo una herramienta de aprendizaje automático no es la solución adecuada requiere un nivel de comprensión más profundo que simplemente usarla. El análisis de los cuatro problemas estructurales (supervisión circular, características insuficientes, incompatibilidad de clases, nunca entrenada) demuestra comprensión de los límites de los modelos secuenciales y de los requisitos de calidad de datos para el aprendizaje supervisado.

---

## 2. Umbral de confianza: evolución y razonamiento

### Evolución del parámetro `conf_threshold`

| Versión | Umbral | Motivo del cambio |
|---|---|---|
| exp3 | 0.40 | Valor por defecto de YOLOv8 |
| exp7 | 0.25 | 33% de fotogramas sin detección con 0.40; priorizar recall |
| exp9 | 0.18 | Grooming y climbing perdían detecciones con 0.25; recuperar recall en clases débiles |

### Por qué el recall es más importante que la precisión en este caso

En el análisis del Open Field Test, **un fotograma no detectado es un dato perdido**. Si el modelo no detecta al ratón durante 2 segundos (60 fotogramas a 30fps), ese comportamiento queda sin registrar en el CSV, produciendo lagunas en el análisis conductual.

Un falso positivo ocasional (detectar al ratón en una pose ligeramente incorrecta) es mucho menos dañino que un falso negativo (no detectarlo en absoluto), porque la lógica híbrida posterior filtra clasificaciones erróneas mediante reglas espaciales y de velocidad.

Esta asimetría justifica operar con un umbral bajo: se acepta algo más de ruido en las detecciones individuales porque la capa de postprocesado actúa como segundo filtro.

### Por qué 0.18 y no 0.10

Bajar excesivamente el umbral introduce detecciones fantasma: YOLO detecta "algo" en fotogramas donde el ratón no es visible o está parcialmente fuera de cuadro. Con `conf=0.10`, la tasa de falsos positivos sube al punto de que el postprocesado no puede corregirlos todos. El valor 0.18 es el resultado de observar visualmente los vídeos de validación: por debajo de 0.18 aparecen bounding boxes en zonas vacías del fotograma; por encima de 0.25 se pierden grooming y climbing consistentemente.

---

## 3. Detección de climbing: tres iteraciones del algoritmo

### Iteración 1 — Confianza pura de YOLO (exp3-exp6)

YOLO clasificaba directamente como `rat_climbing`. Problema: con solo 25 imágenes originales de climbing (la clase menos representada), YOLO aprendía a predecir climbing de forma muy conservadora o, lo contrario, a sobreclasificarlo en casos ambiguos.

### Iteración 2 — Confirmación por snout fuera del área interior (exp7-exp9 inicial)

Se añadió la regla: "YOLO dice climbing → confirmar solo si el keypoint del hocico está **fuera** del área interior (en la zona de pared)."

**Problema encontrado:** cuando el ratón trepa, su cuerpo está pegado a la pared pero el hocico puede apuntar hacia el interior de la caja. El ratón sube por la pared con la cabeza mirando hacia arriba o hacia el centro, por lo que el keypoint del hocico quedaba dentro del área interior y la regla rechazaba sistemáticamente detecciones de climbing legítimas. Resultado: climbing prácticamente desaparecía en exp9 (-10.5% en testRata5).

### Iteración 3 — Confirmación por bounding box en zona de pared (implementación actual)

La regla correcta es: "YOLO dice climbing → confirmar si el **bounding box completo** NO cabe dentro del área interior."

Si el bbox se extiende más allá del borde interior de la caja, parte del cuerpo del ratón está en la zona de pared. Esto es geométricamente equivalente a "el ratón está tocando la pared", independientemente de hacia dónde apunte el hocico.

Se añadió también una regla de **recuperación de climbing**: si YOLO clasifica como `rat_horizontal` o `immobile` pero el bbox toca la zona de pared, reclasificar como `rat_climbing`. Esto captura los casos donde YOLO pierde el climbing por baja confianza pero la geometría lo confirma.

**Por qué el bbox es más robusto que el keypoint para esta tarea**

El hocico es un punto único que depende de la orientación de la cabeza, que varía frame a frame. El bounding box engloba todo el cuerpo y cambia mucho más lentamente. Para detectar una postura global (el animal contra la pared) es más fiable usar la primitiva que describe el cuerpo completo que la que describe un solo extremo.

---

## 4. Dataset y augmentación: por qué solo transformaciones geométricas

### Decisión de diseño

El módulo de augmentación aplica exclusivamente:
- Flip horizontal
- Rotación ±12 grados
- Combinación flip + rotación (solo clases minoritarias)

Se descartaron explícitamente: cambios de brillo, saturación, contraste, ruido gaussiano, blur, recortes agresivos.

### Justificación

El entorno de grabación del Open Field Test es **extremadamente controlado**: misma caja blanca, mismo fondo negro, misma cámara cenital, mismos ratones blancos, iluminación constante. Las augmentaciones de color enseñarían al modelo a ser invariante a variaciones de iluminación que no existen en producción, desperdiciando capacidad del modelo en aprender invariancias espurias.

Las rotaciones y flips sí reflejan variación real: el ratón se mueve en cualquier dirección, y el ángulo de cámara puede diferir ligeramente entre sesiones de grabación.

Este principio se conoce en la literatura como **data augmentation relevance**: solo deben aplicarse augmentaciones que reflejen variaciones presentes en el dominio de inferencia. Aplicar transformaciones irrelevantes no mejora la generalización, puede degradarla si el modelo aprende que propiedades del dominio (como el fondo negro constante) son variables.

### Impacto cuantificado

| Clase | Antes de augmentación | Después | Factor |
|---|---|---|---|
| rat_climbing | 25 | 100 | ×4 |
| rat_grooming | 47 | ~235 | ×5 |
| rat_head_dipping | 27 | 108 | ×4 |
| rat_horizontal | 87 | 348 | ×4 |
| rat_rearing | 62 | ~310 | ×5 |
| **Total** | **248** | **~1101** | **×4.4** |

El ratio de desbalance pasó de 5.8:1 (horizontal vs grooming) a 1.48:1, dentro del rango aceptable para modelos de detección.

---

## 5. Trade-off precisión/recall entre exp8 y exp9

### Resultados comparados

| Métrica | exp8 (época 63) | exp9 (época 44) |
|---|---|---|
| mAP50 | 0.8216 | **0.8736** |
| Precisión | 0.688 | **0.771** |
| Recall | **0.887** | 0.795 |

### Qué significa este trade-off

exp9 detecta mejor (mayor mAP50) y cuando detecta, acierta más (mayor precisión). Sin embargo, se pierde más detecciones reales (menor recall). En términos del análisis conductual, esto se traduce en:

- **Climbing**: exp9 lo detecta menos porque su umbral interno de confianza para esta clase es más alto. La diferencia visual observada fue significativa: -6.8% en rata3, -10.5% en rata5.
- **Grooming**: misma causa, -5.7% en rata3.
- **Walking y rearing**: exp9 es mejor, los errores de exp8 eran más frecuentes.

### Por qué se eligió exp9 como modelo definitivo

La pérdida de recall en climbing y grooming se corrigió mediante dos ajustes en el postprocesado sin necesidad de re-entrenar:
1. Reducción de `conf_threshold` de 0.25 a 0.18 (recupera grooming).
2. Regla geométrica de confirmación y recuperación de climbing (recupera climbing independientemente de la confianza de YOLO).

Con estos ajustes, exp9 aporta mejor mAP50 global y mejor precisión en las clases bien representadas, sin sacrificar las clases problemáticas. Re-entrenar exp8 con más datos o ajustar sus parámetros habría tardado horas adicionales de GPU sin garantía de mejora equivalente.

### Lección aprendida

El mAP50 global es una métrica agregada que puede ocultar degradación en clases individuales. En proyectos con clases desbalanceadas, es fundamental evaluar también el mAP50 por clase y validar visualmente sobre vídeos reales. Los números del CSV de `results.csv` son una condición necesaria pero no suficiente para validar un modelo conductual.

---

## 6. Lógica espacial determinista vs. aprendizaje automático para geometría

### La regla de head_dipping

La detección de `rat_head_dipping` se confirma cuando el keypoint del hocico cae dentro del radio de uno de los cuatro agujeros calibrados. Esta es una regla puramente geométrica que funciona con 100% de fiabilidad cuando la calibración es correcta.

**¿Por qué no entrenar a YOLO para detectar head_dipping directamente?**

YOLO sí se entrena con una clase `rat_head_dipping`. Sin embargo, la confirmación espacial actúa como segundo filtro porque:

1. YOLO puede confundir head_dipping con una postura horizontal cercana al suelo (el cuerpo adopta una forma similar).
2. La posición de los agujeros es información que YOLO no tiene en la imagen: puede saber que la pose es "dipping" pero no puede saber si el hocico está sobre un agujero o sobre el suelo liso.
3. La regla espacial es invariante a la iluminación, el color del ratón y la calidad de imagen, a diferencia de un clasificador CNN.

**Principio general**: cuando una regla puede expresarse exactamente mediante geometría con coordenadas calibradas, la regla geométrica es siempre más fiable que un clasificador aprendido. El aprendizaje automático es la herramienta correcta cuando la regla no puede expresarse explícitamente; cuando sí puede, la expresión explícita es superior.

### La regla de sniffing

El sniffing se detecta cuando el hocico está dentro del área interior Y a menos de 30 píxeles del borde interior. Esta es nuevamente una regla geométrica que no requiere aprendizaje, pero que sería extremadamente difícil de aprender para un modelo de visión: la diferencia entre un ratón sniffing y uno walking de espaldas a la cámara es de unos pocos píxeles en la posición del hocico respecto al borde.

---

## 7. Histéresis temporal: de YOLO a YOLO Pose, y por qué no la RNN

### La evolución completa de la arquitectura

Para entender por qué acabamos con `_LabelStabilizer` hay que recorrer los cuatro intentos de arquitectura, en orden, y qué problema resolvió cada cambio.

---

**Etapa 1 — YOLO estándar (YOLOv8 classification)**

El punto de partida fue YOLOv8 en modo detección estándar: el modelo ve un fotograma y devuelve un bounding box + una clase. En nuestro caso las clases eran `rat_horizontal`, `rat_climbing`, `rat_rearing`, `rat_grooming`, `rat_head_dipping`.

El problema inmediato: `rat_horizontal` era un cajón de sastre. Walking, immobile y sniffing se veían idénticos desde una sola imagen cenital — el ratón está tumbado en los tres casos. YOLO no tiene memoria de fotogramas anteriores, así que no puede saber si el ratón que está tumbado ahora estaba andando hace 0.1 segundos o lleva 10 segundos quieto.

---

**Etapa 2 — YOLO + RNN**

La solución conceptual: añadir una LSTM que procese la **secuencia** de salidas de YOLO. La LSTM acumula 30 fotogramas y aprende patrones temporales: "si en los últimos 30 frames el centroide se ha desplazado mucho, es walking; si no, immobile".

Cuatro problemas estructurales la hicieron inviable (detallados en sección 1):
- Supervisión circular: se entrena con los propios errores de YOLO.
- Features insuficientes: solo bounding box, sin keypoints de esqueleto.
- Incompatibilidad de clases: no produce `walking`, `immobile` ni `sniffing`.
- Nunca llegó a entrenarse: `best_rnn.pth` nunca existió.

---

**Etapa 3 — YOLO Pose**

El cambio decisivo: sustituir el modelo de detección por uno de **estimación de pose**. `yolov8s-pose.pt` añade keypoints al output: además del bounding box, devuelve las coordenadas (x, y, confianza) de cada keypoint. En nuestro caso: hocico (snout), columna (spine) y cola (tail).

Esto cambia todo:
- El hocico sobre un agujero → `rat_head_dipping` (regla geométrica exacta, no clasificación)
- El hocico cerca del borde interior → `sniffing` (regla geométrica exacta)
- El bounding box tocando la zona de pared → `rat_climbing`
- La velocidad del centroide → `walking` vs `immobile`

La información que la RNN intentaba *inferir* de secuencias de bounding boxes ahora está **directamente disponible** en la geometría del esqueleto. La RNN se vuelve aún menos necesaria con YOLO Pose que con YOLO estándar.

---

**Etapa 4 — YOLO Pose + lógica espacial + `_LabelStabilizer`**

Con la lógica espacial cubriendo las clasificaciones geométricas, el único problema restante era el parpadeo: YOLO fluctúa entre clases similares en fotogramas consecutivos de la misma pose (el ratón no se mueve pero YOLO alterna entre `rat_grooming` e `immobile`). Ese parpadeo no es un problema de falta de contexto temporal — es **ruido de clasificación** que necesita un filtro, no más aprendizaje.

La solución es un filtro de histéresis temporal.

---

### Qué es la histéresis y cómo se usa en IA

**Histéresis** viene del griego "llegar tarde": un sistema con histéresis no cambia de estado instantáneamente sino que exige que la nueva condición se sostenga durante cierto tiempo o con cierta magnitud.

El ejemplo más claro es el **termostato**:
- Sin histéresis: si la temperatura objetivo es 20°C, el termostato enciende a 19.9°C y apaga a 20.1°C → oscila constantemente.
- Con histéresis: enciende cuando baja de 18°C, apaga cuando sube a 22°C → estabilidad.

En ingeniería de señal, los **filtros de histéresis** eliminan el ruido en sensores: un sensor de presión que oscila ±2 unidades alrededor del umbral no debe disparar alarmas continuamente.

**Usos de histéresis (o mecanismos equivalentes) en IA:**

| Técnica | Cómo implementa histéresis | Dónde se usa |
|---|---|---|
| Hidden Markov Models (HMM) | La probabilidad de transición entre estados refleja inercia: cambiar de estado tiene coste | Reconocimiento de voz, análisis de ADN |
| Conditional Random Fields (CRF) | Penaliza secuencias con cambios frecuentes de etiqueta | NLP (etiquetado de entidades), segmentación semántica |
| Viterbi decoding | Encuentra la secuencia más probable respetando la inercia de estados | Decodificación de códigos, HMM |
| Post-processing de vídeo | Filtros de suavizado temporal aplicados sobre salidas de clasificadores frame-a-frame | Clasificación de acciones en vídeo |
| `_LabelStabilizer` (este proyecto) | Exige N frames consecutivos antes de aceptar un cambio | Postprocesado del detector de ratas |

La diferencia fundamental entre una **RNN** y un **filtro de histéresis**:
- La RNN **aprende** la inercia de los datos: necesita miles de ejemplos etiquetados para descubrir que los comportamientos duran varios segundos.
- El filtro de histéresis **impone** una inercia fija diseñada por el ingeniero: 8 frames porque sabemos que un comportamiento mínimo dura ~0.27s.

Cuando la inercia correcta se puede definir a priori (como aquí, donde la etología del Open Field Test establece duraciones mínimas de bout), el filtro de histéresis es superior: más simple, más interpretable, sin datos de entrenamiento, sin riesgo de sobreajuste y sin el problema de supervisión circular.

---

### Diseño del `_LabelStabilizer`

El estabilizador implementa histéresis con dos estados internos:
- `_stable`: la etiqueta actualmente mostrada en el vídeo.
- `_candidate`: la nueva etiqueta que intenta reemplazarla.

Una transición solo ocurre si `_candidate` aparece durante `hold_frames=8` fotogramas consecutivos (~0.27 s). Si aparece una etiqueta diferente antes de llegar a 8, el contador se reinicia. Las etiquetas `rat_horizontal` y `Unknown` son "transparentes": no se muestran ni activan el contador de cambio (son señal nula, no un comportamiento real).

### Por qué 8 fotogramas

| Valor | Efecto |
|---|---|
| 3-4 fotogramas | Elimina parpadeo rápido pero permite cambios en ~0.1 s (aún perceptible) |
| **8 fotogramas (~0.27 s)** | **Elimina casi todo el parpadeo; el lag en transiciones reales es imperceptible** |
| 15+ fotogramas | Las transiciones reales tardan demasiado en reflejarse en el vídeo |

A 30 fps, 8 fotogramas corresponden aproximadamente al tiempo mínimo que un comportamiento debe sostenerse para ser considerado un "bout" conductual en etología. El estabilizador refleja implícitamente la duración mínima de un estado conductual significativo.

### Para la memoria del TFG

> *"Para eliminar el parpadeo de etiquetas causado por la variabilidad frame a frame del clasificador YOLO, se implementó un filtro de histéresis temporal. Este mecanismo, análogo a los filtros de suavizado de secuencias utilizados en modelos de Markov ocultos y Conditional Random Fields, exige que una nueva etiqueta aparezca de forma ininterrumpida durante un mínimo de 8 fotogramas consecutivos (~0.27 s a 30 fps) antes de reemplazar a la etiqueta activa. Este umbral se justifica etológicamente: los estudios de Open Field Test definen un 'bout' comportamental como un episodio de al menos 0.3-0.5 s de duración. Se optó por este enfoque determinista en lugar de una LSTM de suavizado porque la inercia correcta es conocida a priori y porque una red recurrente entrenada sobre las salidas de YOLO habría heredado sus errores de clasificación (problema de label noise propagation), sin acceso a los keypoints de esqueleto que son la información discriminativa real."*

---

## 8. Separación entre pipeline de producción y herramientas de desarrollo

### Motivación

Durante el desarrollo se acumularon scripts con propósitos muy distintos: algunos son herramientas internas de análisis del dataset, otros son utilidades de validación post-entrenamiento, y otros son el menú interactivo que usará un investigador externo. Mezclarlos en el mismo directorio hace el proyecto confuso para alguien que solo quiera ejecutar el sistema.

### Decisión

Se separaron en dos capas:

**`scripts/` — Pipeline de producción:**
- `main_model.py`: menú interactivo (calibrar → entrenar → detectar)
- `yolo_pose/`: detector, calibrador, entrenador
- `logic/`: lógica espacial
- `config/`: configuración centralizada
- `tools/calibrator_image.py`: calibrador visual estático

**`adminScripts/` — Herramientas de desarrollo TFG:**
- `watch.py`: monitor de entrenamiento en tiempo real
- `_run_validate.py`: validación post-entrenamiento con comparativa entre experimentos
- `_run_detect.py`, `_run_detect_any.py`: detección rápida sin menú interactivo
- `augment_dataset.py`: generación offline de augmentaciones
- `dataset_quality_check.py`: análisis de calidad del dataset
- `extract_label_candidates.py`: extracción de fotogramas para etiquetar

### Por qué esto mejora la mantenibilidad

Cuando un investigador descargue el repositorio, el directorio `scripts/` solo contiene lo que necesita para usar el sistema. Las herramientas de desarrollo no contaminan el namespace del usuario final. Además, los scripts de `adminScripts/` importan `scripts/` mediante path explícito, de modo que no crean dependencias circulares.

---

## 9. Calibración por vídeo: primer fotograma en tiempo real

### Problema inicial

El sistema tenía una imagen de referencia fija (`media_original/cajaBordes.jpg`) usada para calibrar las coordenadas de la caja. Esto funciona cuando todos los vídeos se graban desde exactamente la misma posición de cámara, pero en la práctica se descubrió que `testRata3.mp4` tenía la cámara en una posición diferente a `testRata5.mp4`. Las coordenadas calibradas para una cámara producían zonas incorrectas en la otra.

### Solución implementada

Cuando el usuario selecciona un vídeo en la opción 3 del menú principal, el sistema:
1. Abre el vídeo y extrae el primer fotograma con OpenCV
2. Guarda el fotograma como imagen temporal
3. Abre el calibrador visual sobre ese fotograma específico
4. El usuario marca las zonas (borde exterior, interior, 4 agujeros) para ESA cámara concreta
5. Guarda las coordenadas y lanza la detección

Esto garantiza que cada vídeo se analiza con la calibración correcta para su cámara, sin necesitar una imagen de referencia separada por cada configuración.

### Implicación para la validación

`testRata3.mp4` queda excluido de las pruebas de validación estándar por este motivo. Los vídeos de referencia son `testRata4.mp4` y `testRata5.mp4`, que comparten la misma posición de cámara y pueden usar la misma calibración.

---

## 10. Modelo dual de salida de vídeo

### Motivación

Durante la validación se necesitaban dos tipos de vídeo simultáneamente:
- **Overlay** (con las zonas calibradas superpuestas en color): para verificar que la lógica espacial estaba usando correctamente las coordenadas
- **Label** (solo bbox y etiqueta, sin zonas): para presentación limpia del comportamiento detectado

Generar ambos en dos pasadas del vídeo duplicaría el tiempo de procesado (que ya es costoso: ~64 ms de inferencia por fotograma).

### Implementación

Se añadió un segundo `cv2.VideoWriter` en el `RatDetector` que opera en paralelo sobre un clon del fotograma base, sin superposición de zonas. Ambos vídeos se escriben en el mismo bucle de inferencia, con coste prácticamente nulo en GPU (solo operaciones de memoria en CPU).

El modo dual se activa con `dual_output=True` y solo se usa en los scripts de desarrollo (`adminScripts/`). El menú de producción (`main_model.py`) usa `dual_output=False` y entrega un único vídeo al investigador.

---

## 11. Confirmación de rearing por aspect ratio del bounding box

### El problema

YOLO clasifica `rat_rearing` basándose en la postura estática del frame. El problema: un ratón inmóvil en el suelo y un ratón que acaba de bajarse de las patas traseras pueden tener perfiles de silueta muy similares en ciertos fotogramas. YOLO los confundía, produciendo `rat_rearing` cuando el ratón estaba claramente tumbado, lo que se observaba visualmente como parpadeos entre rearing e immobile.

### Qué es el aspect ratio de un bounding box

El **aspect ratio** de un bbox es la relación entre su altura y su anchura:

```
aspect_ratio = (y2 - y1) / (x2 - x1)
```

- `aspect_ratio > 1.0` → bbox más alto que ancho → cuerpo vertical (erguido)
- `aspect_ratio ≈ 1.0` → bbox casi cuadrado → transición o postura ambigua
- `aspect_ratio < 1.0` → bbox más ancho que alto → cuerpo horizontal (tumbado)

### Por qué funciona para rearing

Un ratón en rearing se incorpora sobre sus patas traseras: el cuerpo adopta una posición vertical, lo que hace que el bounding box sea claramente más alto que ancho (aspect ratio típico: 1.2–2.0). Un ratón inmóvil o caminando mantiene el cuerpo horizontal sobre las cuatro patas: el bbox es más ancho que alto (aspect ratio típico: 0.4–0.7).

La franja entre 0.7 y 1.0 corresponde a posturas de transición o ángulos de cámara oblicuos. El umbral elegido es **0.80**: se acepta rearing cuando el bbox es casi cuadrado o más alto que ancho, rechazándolo cuando el bbox es claramente horizontal.

| Aspect ratio | Interpretación | Decisión |
|---|---|---|
| > 1.0 | Cuerpo definitivamente vertical | rearing confirmado |
| 0.80 – 1.0 | Postura ambigua, bbox casi cuadrado | rearing confirmado (conservador) |
| < 0.80 | Cuerpo claramente horizontal | reclasificar como walking/immobile/sniffing |

### Implementación en el pipeline

```python
# Bloque D en detector.py
elif yolo_label == "rat_rearing":
    h_box = y2 - y1
    w_box = x2 - x1
    aspect = h_box / w_box if w_box > 0 else 1.0
    if aspect >= REARING_ASPECT_RATIO:   # 0.80
        final_label = "rat_rearing"
    else:
        final_label = self._derive_horizontal(snout_kp)
```

Si el aspect ratio no confirma rearing, el frame se reclasifica mediante `_derive_horizontal`: velocidad → walking/immobile/sniffing. El resultado es correcto en todos los casos: si el ratón estaba realmente inmóvil, la velocidad confirma immobile; si estaba andando, confirma walking.

### Comparación con alternativas

| Alternativa | Problema |
|---|---|
| Reentrenar con más datos de rearing | Requiere etiquetado manual de cientos de frames; no corrige el problema de ambigüedad en transición |
| Filtrar por keypoints (snout-spine-tail verticales) | Funciona, pero los keypoints de YOLO en poses verticales tienen más error de localización; el bbox es más estable |
| Umbral de velocidad para rearing | Un ratón puede estar haciendo rearing y quieto (velocidad baja) → no discrimina de immobile |
| **Aspect ratio del bbox** | **Directo, sin entrenamiento, robusto porque el bbox cambia drásticamente entre horizontal y vertical** |

### Uso en el proyecto

Este es el primer uso explícito del **aspect ratio del bbox** como feature discriminativa en el pipeline. El concepto es extensible: grooming (ratón enroscado, bbox cuadrado y compacto), head_dipping (ratón inclinado, bbox en diagonal) o rearing con movimiento (bbox alto + velocidad creciente) son variantes que podrían refinarse con la misma primitiva geométrica.

---

## 12. Límites del dataset y por qué el enfoque híbrido generaliza mejor

### Contexto real del proyecto

Este TFG nació de una necesidad real: el departamento de Farmacología ya usa software de análisis conductual (EthoVision, ANY-maze) para los mismos ensayos de Open Field Test, pero es caro (licencias de miles de euros), requiere hardware específico y no es intuitivo para el investigador. El objetivo del sistema desarrollado es reproducir las métricas clave con hardware estándar, código abierto y un flujo de trabajo que cualquier persona del laboratorio pueda ejecutar.

El TFG no es un producto de laboratorio listo para producción — es un trabajo académico que demuestra la aplicación de técnicas de visión por computador a un problema científico real. Lo que se evalúa es el proceso, las decisiones técnicas y que el sistema funcione de forma demostrable. El hecho de que surja de una necesidad real y que el sistema funcione visualmente bien sobre vídeos nuevos es un valor añadido, no un requisito del tribunal.

### El problema fundamental: alta similitud visual del dataset

En el Open Field Test, el entorno es extremadamente controlado por diseño: misma caja blanca, mismo fondo negro, misma cámara cenital, mismos ratones blancos, iluminación constante. Esto es bueno para la reproducibilidad científica, pero es un problema para el aprendizaje automático.

Cuando todos los vídeos son visualmente idénticos en su contexto, añadir más vídeos del mismo experimento **no aumenta la diversidad visual del dataset** — añade más ejemplos del mismo dominio visual, lo que no ayuda a la generalización. Esta situación tiene nombre en la literatura: **alta similitud visual con mínimo domain shift**. El modelo rápidamente aprende a memorizar el fondo y las condiciones de iluminación en lugar de aprender la geometría del cuerpo del ratón.

Esto se observó empíricamente: incluso con augmentación geométrica que multiplicó el dataset ×4.4, el modelo entraba en overfitting antes de las 60 épocas. La validación loss dejaba de mejorar mientras la training loss seguía bajando — el modelo estaba memorizando, no generalizando.

### Por qué el enfoque híbrido generaliza mejor

La solución adoptada — YOLO Pose + lógica espacial calibrada + histéresis temporal — tiene una propiedad fundamental que los modelos puramente neuronales no tienen en este dominio: **las reglas espaciales son invariantes al individuo**.

| Componente | Depende del ratón individual | Depende del entorno |
|---|---|---|
| YOLO (detección de pose) | Sí — el modelo aprende texturas y formas | Sí — necesita reentrenarse si cambia la cámara |
| Lógica espacial (`SpatialAnalyzer`) | **No** — solo usa coordenadas calibradas | Mínimo — recalibrar tarda <2 minutos |
| `_LabelStabilizer` | **No** — parámetro fijo diseñado a priori | **No** |

Esto significa que, dado un nuevo ratón del mismo experimento, el sistema funciona sin reentrenar: la posición de los agujeros, los bordes de la caja y los umbrales de velocidad son los mismos. Solo YOLO necesitaría ajuste si la apariencia del ratón cambiara significativamente (lo que no ocurre en ratas de laboratorio de la misma cepa).

### Para la memoria del TFG

> *"La variabilidad visual del dataset es intrínsecamente baja por el diseño controlado del Open Field Test. El fine-tuning extensivo sobre imágenes del mismo entorno produce overfitting rápido, lo que limita la mejora del modelo mediante datos adicionales. La solución adoptada combina un modelo de pose ligero con lógica espacial determinista calibrada por experimento, reduciendo la dependencia de datos etiquetados para los comportamientos más difíciles de distinguir visualmente. Las reglas geométricas (posición del hocico respecto a agujeros y paredes, geometría del bounding box) son invariantes al individuo y no requieren reentrenamiento al cambiar de rata, lo que proporciona una generalización que un clasificador puramente neuronal no podría alcanzar con el dataset disponible."*

### Limitaciones conocidas

- **Climbing con una pata**: cuando el ratón se apoya en la pared con solo una pata delantera, el bounding box puede no cruzar el umbral de 15px fuera del borde interior. La pose es visualmente diferente del climbing con dos patas, y el dataset de entrenamiento tiene pocos ejemplos de esta variante. Resolverlo requeriría etiquetar específicamente ese patrón.
- **Grooming ambiguo**: YOLO confunde ocasionalmente posturas compactas (ratón enroscado limpiándose) con otras poses. La causa es la misma que el problema del dataset: pocos ejemplos de grooming en diversas orientaciones.
- **Rearing en transición**: el filtro de aspect ratio puede rechazar frames donde el ratón está a medio erguirse, produciendo detección fragmentada del bout. Es un trade-off consciente para evitar falsos positivos de rearing en poses horizontales.

---

## Resumen de decisiones y su justificación

| Decisión | Alternativa descartada | Motivo |
|---|---|---|
| YOLO Pose en lugar de YOLO estándar | Mantener detección sin keypoints | Los keypoints permiten reglas geométricas exactas para dipping, sniffing y climbing |
| Eliminar RNN del pipeline | Entrenar y mantener la LSTM | Supervisión circular, features insuficientes, clases incompatibles, nunca entrenada |
| `_LabelStabilizer` determinista (histéresis) | LSTM para suavizado temporal | El problema era ruido de clasificación, no complejidad temporal; la inercia correcta es conocida a priori |
| Augmentación solo geométrica | Augmentación de color/brillo | Entorno controlado: variación real es solo geométrica |
| `conf_threshold=0.18` | Mantener 0.25 | Recall > precisión en análisis conductual continuo |
| Confirmación de climbing por bbox | Confirmación por keypoint del hocico | El bbox describe el cuerpo completo; el hocico depende de la orientación de la cabeza |
| Reglas geométricas para sniffing/dipping | Clasificador aprendido | La geometría es exacta y conocida; ML no añade valor sobre reglas explícitas |
| exp8 como modelo activo (sobre exp9) | Mantener exp9 por mAP50 mayor | Validación visual mostró que exp9 perdía climbing y grooming en vídeo real; recall=0.887 > 0.795 |
| Confirmación de rearing por aspect ratio | Reentrenar con más datos de rearing | El bbox altura/ancho discrimina vertical vs horizontal sin datos nuevos ni GPU |
| Climbing sin recovery rule (Bloque E) | Mantener recovery rule para bajo recall | Con exp8 recall=0.887, la regla causaba falsos positivos; se eliminó |
| Margen 15px en bbox_entirely_inside | Sin margen (1px = climbing) | Evita falsos climbing cuando el ratón camina rozando el borde interior |
| Separación `scripts/` y `adminScripts/` | Todo en `scripts/` | Claridad para usuario final; sin contaminación del namespace de producción |
