# Pareto-MC — diario de desarrollo

## Declaración previa al entrenamiento (21 de octubre de 2026)

- Rama `feat/pareto-mc`, desde `review/robust-mvdr` (`0613abe`); PR #1 abierto.
- Entrenamiento: sub-01, sub-02, sub-03; validación: sub-04; sub-05 reservado.
- Cambio explícito frente a Stage B: se permiten las cinco condiciones y todos
  los SNR (30, 25, 20, 15 dB) de entrenamiento y validación.
- Inferencia: solo EMG crudo, calibración inicial y estado pasado. El ángulo
  entra exclusivamente en supervisión y evaluación angular.
- Pareto publicado utiliza preprocesado offline. C0 usará el preprocesado causal
  con estado del commit fijado y se contrastará con una nueva ejecución causal:
  referencia **Pareto-rerun**. No se atribuirán sus diferencias a C1–C4.
- Las puertas se deciden solo en validación. Máximo tres iteraciones por nivel,
  con como máximo un cambio estructural tras un fallo; si vuelve a fallar, parada.
- Resultados, protocolos y predicciones nuevos en `outputs/pareto_mc`; los
  experimentos anteriores permanecen intactos. Sin push ni PR sin autorización.

## L0 — hipótesis registrada

Un bucle propio que conserva las operaciones, el orden, la precisión y el estado
de Pareto reproduce su ejecución causal. Tres registros de sub-01: staircase,
ramp10s y ramp5s, 30 dB. Puerta: ≥99,9 % de muestras iguales y diferencia máxima
de RoA por unidad ≤0,01 puntos. Se guardan y sellan las predicciones antes de
abrir las etiquetas para comprobar RoA. Resultado pendiente de ejecución.

### L0 — resultado y decisión

Puerta **superada** en los tres registros: 100 % de muestras iguales y diferencia
máxima de RoA 0,0000 puntos, tanto frente al método original como al Pareto
publicado. RoA 88,7598 / 61,1219 / 51,0367 % (solo estas tres pruebas sub-01).
Medianas 38,0525 / 38,4831 / 38,1205 ms; p95 46,7016 / 47,2096 / 46,5453 ms.
El criterio final de latencia de 22 ms **no está satisfecho**.

Se conserva «Pareto-rerun», nombre preregistrado aunque también coincida con
las salidas publicadas. Corrección de la interpretación inicial: el filtro
offline de la referencia también usa `sosfilt` causal; la media global de
preparación se cancela por centrado de lote. El bucle nuevo evita esa dependencia.

Incidencias: primer protocolo conservado en `outputs/pareto_mc/L0` fallido por
`from_calibration` sin `cbss_config`; corregido usando el constructor explícito
con todos los tensores de calibración. Nueva ejecución completa e inmutable en
`outputs/pareto_mc/development_v1/L0`. El constructor original emite errores MKL
SLASWP al calcular estadísticas KL; no afecta al paso fijo y se conserva la
coincidencia exacta. Ruff se ejecuta desde `.venv-windows`: `uv run` encontró
una `.venv` local de otra plataforma y falló antes de ejecutar Ruff.

Fase 0 fallida: intento real de contenedor oficial, Docker Linux apagado;
sin Singularity/Apptainer en Ubuntu. Sin datos confirmatorios. Se conserva el
informe `outputs/pareto_mc/development_v1/phase0.json`. No se sustituye FDSI
por el generador analítico de FIFO.

## L1 — hipótesis previa al entrenamiento

El clasificador residual 22→8→1, reentrenado con candidatos agregados de sus
propias trayectorias, puede conservar la ganancia abierta al realimentar SV y
centroides. La calibración inicial permanece Pareto, los retardos de etiquetas
se estiman solo en sus cinco segundos y la tolerancia es 2 ms.
C1 se ajusta antes de C4 por NLL en los sujetos autorizados; su velocidad
estimada entra en C4. Memoria y controlador no modifican filtros en L1.
Puerta: RoA global de validación ≥ verosimilitud abierta y ramp5s ≥ abierta+1 pp.
No se procede a L2 si falla la puerta tras el único cambio estructural permitido.
