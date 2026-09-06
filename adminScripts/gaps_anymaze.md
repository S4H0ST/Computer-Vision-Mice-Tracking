# Gaps vs ANY-maze — Qué hace, qué falta, qué supera

> Basado en el email de Ana (Farmacología) y el análisis del pipeline actual.
> Último update: septiembre 2026

---

## Lo que hace ANY-maze (según Ana)

| Métrica | Descripción | ¿Lo hace nuestro sistema? |
|---|---|---|
| Velocidad de movimiento | Velocidad media del animal a lo largo del vídeo | ✅ En cm/s si se calibra la caja, si no en px/s |
| Metros en zona central | Distancia recorrida en los 4 cuadrados entre agujeros | ⚠️ Distancia total sí, pero **no dividida por zona** |
| Metros en periferia | Distancia recorrida en la zona exterior a los agujeros | ⚠️ Mismo problema |
| Zona central = entre agujeros | Los 4 cuadrados delimitados por la posición de los 4 agujeros | ⚠️ Nuestro sistema usa `inner_limits` (borde interior de la caja), **no la zona entre agujeros** |

---

## Lo que ANY-maze NO hace (pero nuestro sistema sí)

| Comportamiento | Descripción | Estado |
|---|---|---|
| Head dipping | Nº de bouts, duración total, latencia al primer dip, dips por cuarto (habituación) | ✅ Implementado |
| Grooming | Bouts/min, duración total, duración media por bout | ✅ Implementado |
| Rearing | Detección por aspect ratio del bbox | ✅ Implementado |
| Sniffing de pared | Exploración olfativa del perímetro (hocico a <30px del borde interior) | ✅ Implementado |
| Climbing | Trepado a la pared (bbox cruza el borde interior) | ✅ Implementado |
| Mapa de trayectoria del hocico | Recorrido del snout sobre la caja (relevante en holeboard) | ✅ Implementado |
| Presupuesto de tiempo conductual | % tiempo por comportamiento + bouts + duración media | ✅ Implementado |

---

## Gaps reales que quedan por cerrar

### Gap 1 — Distancia y velocidad en unidades reales ✅ CERRADO
**Estado:** implementado. Al calibrar, el menú pregunta el ancho y alto de la caja en cm.
Si se introduce, el Excel muestra metros y cm/s. Si no, muestra píxeles.

### Gap 2 — Distancia dividida por zona (centro vs periferia)
**Descripción:** ANY-maze reporta por separado los metros en zona central y en periferia.
Nuestro sistema calcula la distancia total pero no la acumula por zona.
**Cómo implementarlo:** en el bucle de cálculo de distancia (tail frame a frame),
comprobar si el centroide está en la zona central o periférica y acumular por separado.
**Prioridad:** media. Ana ya tiene este dato de ANY-maze.

### Gap 3 — Definición de zona central
**Descripción:** Ana define zona central como los 4 cuadrados delimitados por los agujeros.
Nuestro sistema usa `inner_limits` (el interior de la caja delimitado por las paredes),
que es un área mucho mayor.
**Cómo implementarlo:** calcular un rectángulo a partir de las coordenadas de los 4 agujeros
(min_x, min_y, max_x, max_y de los agujeros) y usarlo como definición de zona central.
**Prioridad:** alta si se quiere comparar con los datos de ANY-maze de Ana.

### Gap 4 — Métricas por agujero individual
**Descripción:** ANY-maze no las tiene, pero para un holeboard sería valioso saber
cuántos dips hizo el animal en cada agujero por separado (preferencia lateral o espacial).
**Cómo implementarlo:** en `check_dipping`, retornar el índice del agujero activado
en lugar de solo True/False, y acumular dips por índice de agujero.
**Prioridad:** alta. Es una ventaja real frente a ANY-maze que Ana valoraría.

### Gap 5 — Mapa de calor
**Descripción:** mostrar la densidad de tiempo del hocico en cada zona de la caja.
Mucho más legible que la trayectoria de líneas para identificar zonas preferidas.
**Cómo implementarlo:** acumular posiciones del snout en un array 2D,
aplicar GaussianBlur y un colormap (COLORMAP_HOT).
**Prioridad:** media. Muy buena para la presentación visual del TFG.

---

## Resumen de posición frente a ANY-maze

```
ANY-maze:       tracking de posición + zonas + distancia/velocidad
Nuestro sistema: tracking + zonas + distancia/velocidad + CLASIFICACIÓN DE CONDUCTA

La clasificación de conducta (head-dipping, grooming, rearing, sniffing, climbing)
es el aporte único del TFG. Ana no puede obtenerlo de ANY-maze.
```

---

## Frase para la memoria / reunión con el tutor

> "El sistema reproduce las métricas de movimiento y zonificación de ANY-maze
> e incorpora la clasificación automática de posturas conductuales que el software
> comercial no puede cuantificar: head-dipping (con latencia y habituación por cuartos),
> grooming, rearing, sniffing de pared y climbing. Esto cubre directamente
> la limitación que el laboratorio de Farmacología identificó en su flujo de trabajo actual."
