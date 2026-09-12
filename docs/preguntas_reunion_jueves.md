# Reunión jueves — Validación TFG
**Asistentes:** Tutor TFG + Ana (Farmacología)  
**Objetivo:** Cerrar lo que queda antes de la memoria

---

## Para Ana (Farmacología)

### Validación de comportamientos ← prioritario
- ¿Tienes vídeos ya anotados manualmente con los comportamientos (grooming, rearing, head dipping, etc.)? Aunque sea uno solo serviría para calcular precisión del sistema contra ground truth.
- Si no tienes anotaciones previas, ¿podrías anotar uno de los dos vídeos de test que ya tenemos? No hace falta frame a frame — con timestamps de eventos es suficiente.

### Criterios de comportamiento
- ¿Los comportamientos que detecta el sistema (grooming, rearing, head dipping, sniffing, walking, immobile) cubren los que usas en tus experimentos, o falta alguno relevante?
- ¿Hay algún comportamiento que el sistema clasifique mal de forma consistente en los vídeos que has visto?

### Datos de experimentos
- ¿Tienes vídeos de grupo control vs. grupo con fármaco? Aunque sean experimentos anteriores, si el sistema puede mostrar diferencias estadísticas entre grupos eso refuerza mucho la validación científica.
- ¿Cuál es la duración estándar de un experimento en tu lab? ¿Va bien con esa duración?

### Utilidad práctica
- De los datos que genera el sistema (tiempo por zona, comportamientos, trayectoria, mapa de calor, Excel OFT), ¿cuáles usarías realmente en un análisis? ¿Falta alguna métrica que AnyMaze sí te daba?
- ¿La calibración de zonas (pared, centro, agujeros) se adapta a vuestro open field real, o hay que ajustar algo?

---

## Para el tutor TFG

### Estado del TFG
- ¿Qué falta técnicamente para dar el trabajo por cerrado antes de la memoria?
- El modelo activo es exp9 (mAP50=0.874). ¿Es suficiente o conviene un ajuste de umbral para climbing/grooming antes de entregar?

### Memoria
- ¿Qué extensión espera el tribunal para la memoria?
- ¿La validación del sistema puede ser la comparación del sistema contra anotación manual de Ana, o necesita algo más formal?
- ¿Hay que incluir el código en anexos, o con el repositorio es suficiente?
- ¿Se espera un apartado de evaluación cuantitativa de los comportamientos, o las métricas de YOLO (mAP, precisión, recall) cubren la parte de evaluación?

### Defensa
- ¿El formato es demo en vivo + presentación, o solo presentación?
- ¿Cuánto tiempo tiene la defensa y cuánto el turno de preguntas?

---

## Para mostrar en la demo

- Pipeline completo: vídeo entrada → calibración de zonas → detección → CSV + vídeo anotado
- Mapa de calor y trayectoria
- GUI: las 4 páginas (calibración, detección, resultados, estadísticas)
- Cámara en vivo si hay cámara disponible
- Los 2 vídeos de test con sus resultados

**Preguntar:** ¿Preferís ver el vídeo con overlay (zonas pintadas + bbox + label) o el vídeo limpio (solo bbox + label), o ambos? Según respuesta, ajustar qué se genera por defecto.

---

## Preguntas frecuentes en la defensa — respuestas preparadas

**"¿Necesita GPU? ¿Qué pasa en un equipo sin tarjeta gráfica?"**

> El sistema detecta automáticamente si hay GPU disponible al arrancar. En CPU funciona correctamente para el modo vídeo — solo es más lento (3–8 fps de procesamiento en lugar de tiempo real). Para el uso habitual en un laboratorio de farmacología, el flujo es grabar el experimento y analizarlo después, no en tiempo real, así que CPU es suficiente. Para tiempo real con cámara en vivo sí se recomienda GPU. Como línea futura, el modelo se puede exportar a ONNX o OpenVINO para mejorar el rendimiento en CPU sin GPU.

**"¿Cómo usa la GPU exactamente? ¿Hay código CUDA manual?"**

> No hay kernels CUDA escritos a mano. El código pasa `device="0"` a `YOLO.predict()` de Ultralytics, que internamente usa PyTorch para mover el modelo y los tensores a la VRAM y ejecutar las convoluciones a través de cuDNN. Es la forma estándar y recomendada — escribir kernels CUDA a mano para esto sería reinventar lo que cuDNN ya optimiza para arquitecturas Nvidia.

**"¿Por qué desktop y no web/nube?"**

> El caso de uso es un laboratorio con un ordenador fijo y experimentos con vídeos propios (datos sensibles). Una app de escritorio encaja mejor que una solución cloud: funciona sin internet, no requiere subir vídeos a servidores externos, y la calibración interactiva sobre el frame es mucho más natural en una app nativa que en un navegador. Como trabajo futuro se podría desplegar en un servidor interno del laboratorio con acceso en red local.

---

## Decisiones que quedan abiertas tras la reunión

- [ ] ¿Ana tiene vídeos anotados? → decide si se hace validación cuantitativa antes de la memoria
- [ ] ¿Re-entrenar con ajuste de climbing, o exp9 es definitivo?
- [ ] ¿Qué métricas añadir al CSV/Excel según feedback de Ana?
- [ ] Fecha límite entrega memoria
