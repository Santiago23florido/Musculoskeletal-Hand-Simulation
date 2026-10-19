# Diagnostico robust_mvdr en CPU

Resultado exploratorio sobre los mismos 80 registros y 591 instancias. Los sujetos sub-02 a sub-05 ya se evaluaron en experimentos anteriores: este resultado es diagnostico, no confirmatorio.

Todos los valores son porcentajes; las diferencias se expresan en puntos porcentuales.

| Metodo | RoA medio | Mediana | ramp5s | Precision | Recall | moving | reference_rest | Delta Pareto | Delta Stage B static |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pareto | 80.8885 | 95.2933 | 65.8098 | 91.4710 | 83.1556 | 79.6205 | 87.4589 | 0.0000 | 35.3330 |
| stage_b_static_analytic | 45.5555 | 43.0939 | 39.1654 | 91.3866 | 47.8524 | 39.9740 | 96.9062 | -35.3330 | 0.0000 |
| stage_b_moca_k | 36.9673 | 36.8194 | 33.3412 | 92.0868 | 39.0262 | 30.4857 | 96.8127 | -43.9212 | -8.5882 |
| stage_b_moca_b | 37.2580 | 37.3958 | 33.7790 | 92.1682 | 39.3038 | 30.8334 | 96.7156 | -43.6304 | -8.2974 |
| static_subtract | 5.2154 | 5.5024 | 5.4980 | 99.6437 | 5.2164 | 0.1352 | 0.0943 | -75.6730 | -40.3401 |
| static_subtract_strong | 5.3006 | 5.5156 | 5.5581 | 99.4939 | 5.3024 | 0.1996 | 0.6782 | -75.5878 | -40.2548 |
| static_loaded | 5.2890 | 5.5371 | 5.5973 | 99.9455 | 5.2892 | 0.1815 | 0.6502 | -75.5994 | -40.2664 |
| static_fifo_mvdr | 5.2092 | 5.5000 | 5.4867 | 99.9562 | 5.2093 | 0.1272 | 0.0432 | -75.6792 | -40.3462 |
| moca_k_subtract | 5.2288 | 5.5055 | 5.5116 | 99.5805 | 5.2301 | 0.1504 | 0.1020 | -75.6597 | -40.3267 |
| moca_b_subtract | 5.2325 | 5.5055 | 5.5181 | 99.5632 | 5.2338 | 0.1522 | 0.1495 | -75.6559 | -40.3230 |

## Comparaciones preregistradas

1. **No supera al estático anterior.** Pierde 40,3401 puntos de RoA y 42,6360 de recall; también empeora en movimiento y en ramp5s. En reposo de referencia cae de 96,9062% a 0,0943%. El fallo afecta también a la postura de calibración, por lo que no puede describirse solo como un déficit durante movimiento.
2. **La sustracción y el control FIFO son casi iguales descriptivamente:** solo +0,0062 puntos de RoA. No se fijó un margen formal de equivalencia. El colapso compartido del control impide atribuir de manera limpia el resultado a la autocancelación; esta ejecución no respalda que la sustracción propuesta resuelva el déficit de FDSI.
3. **El morph aporta una diferencia mínima:** +0,0134 puntos para K y +0,0171 para B. La razón target-reliability es 30,9224, lejos de 1. Esta auditoría no respalda la hipótesis de ausencia de señal postural distinguible del ruido, y tampoco demuestra que el morph pueda representarla o generalizar.
4. **Persisten brechas de 75,6730 puntos frente a Pareto y 78,2976 frente a la verosimilitud de disparos.** Ninguna de las seis variantes corrige el déficit. El método primario empeora las 591 instancias; no se reajustó ningún parámetro después de observarlo.

Una corrección adicional de lectura del código: Stage B devuelve `R^-1 h / sqrt(h^T R^-1 h)`, mientras `robust_filters` devuelve `R^-1 h / (h^T R^-1 h)` cuando rho=0. Comparten dirección, pero difieren en amplitud. Por tanto, `static_fifo_mvdr` no es una reproducción numéricamente idéntica del decodificador anterior. Esta diferencia está comprobada en el código; su responsabilidad causal en el colapso no se ha aislado con otro experimento. Se preservan los resultados y el código registrado.

Las diferencias exactas registradas son:

- static_subtract - stage_b_static_analytic, RoA: -40.3401 pp.
- static_subtract - stage_b_static_analytic, recall: -42.6360 pp.
- La misma diferencia en movimiento: -39.8387 pp.
- La misma diferencia en ramp5s: -33.6673 pp.
- static_subtract - static_fifo_mvdr: +0.0062 pp de RoA y +0.0071 pp de recall. El README no fijo un margen formal de equivalencia para la expresion aproximadamente igual.
- moca_k_subtract - static_subtract: +0.0134 pp de RoA.
- moca_b_subtract - static_subtract: +0.0171 pp de RoA.
- Razon target-reliability en bins desplazados: 30.922377. Esta razon esta lejos de 1; no prueba que el morph aprenda o generalice esa variacion.
- static_subtract - Pareto (80.8885%): -75.6730 pp.
- static_subtract - spike likelihood (83.5130%): -78.2976 pp.

## Auditorias

Medias sobre cinco sujetos:

| Condicion | Angulo 5 s | Angulo 10 s | Maximo absoluto 0-10 s | Fraccion en movimiento |
| --- | ---: | ---: | ---: | ---: |
| staircase | 0.000000 | 4.999756 | 4.999756 | 0.500000 |
| triangular-ramp40s | 0.000000 | 4.999573 | 4.999573 | 0.500000 |
| triangular-ramp20s | 0.000000 | 9.999268 | 9.999268 | 0.502490 |
| triangular-ramp10s | 0.000000 | 19.999023 | 19.999023 | 0.503711 |
| triangular-ramp5s | 0.000000 | 40.000000 | 40.000000 | 0.504053 |

Target reliability (solo sub-01):

```json
{
  "raw_target_noise_nmse": 0.06034950539469719,
  "raw_change_nmse": 0.5876498222351074,
  "projected_target_noise_nmse": 0.03469512239098549,
  "projected_change_nmse": 0.9899304509162903,
  "projected_signal_to_noise_ratio_moved_bins": 30.922377392072914
}
```

## Tiempo y desviaciones

Registro a informe: 8157.659 segundos (2 h 15 min 57,7 s). Hasta dos procesos concurrentes, cuatro hilos por proceso, exclusivamente CPU.

| Metodo | Mediana de median_core_batch_ms sobre 80 registros |
| --- | ---: |
| pareto | No medido: cache publicada |
| stage_b_static_analytic | 10.5628 |
| stage_b_moca_k | 11.0017 |
| stage_b_moca_b | 10.9178 |
| static_subtract | 16.9971 |
| static_subtract_strong | 15.5373 |
| static_loaded | 5.3935 |
| static_fifo_mvdr | 5.5059 |
| moca_k_subtract | 15.3014 |
| moca_b_subtract | 16.3176 |

Los tiempos excluyen proyeccion, preparacion de caracteristicas y disco; no son latencia completa de un sistema en vivo. Los tiempos de referencia Stage B proceden de su ejecucion anterior.

- Guarded CPU checkpoint loader permits only the optional-PDF metadata AST change.
- Two concurrent subject processes, four threads each, to limit memory usage.
- Previously completed audits reused; their code and inputs are unchanged.
- Timing observation: A long interval without tool response/progress was observed. sub-05 triangular-ramp10s at 30dB reported 2722.2 wall-clock seconds. Processes continued, no computation exception appeared, and approximately 10 GB RAM remained free. Cause not determined; preserve raw timings and distinguish elapsed wall time from median core-batch measurements.
- Dos registros conservaron tiempos de reloj excepcionalmente largos: sub-04/ramp5s/25dB, 2735,2214 s; sub-05/ramp10s/30dB, 2722,1633 s. El total incluye ese intervalo; no se descartaron ni corrigieron sus mediciones. La causa externa no está determinada.
- Se conserva el protocolo y los checkpoints originales. La compatibilidad permite solo el cambio de metadatos del PDF; todos los demas nodos AST deben coincidir.
- No se entreno, no se ajustaron metodos ni umbrales, y no se editaron predicciones ni resultados puntuados.
- Se verificaron los hashes de 57 fuentes anteriores y los tamanos/fechas de 1449 archivos existentes de Stage B; los JSON, pesos y fuentes archivadas tambien conservaron sus hashes.
- Validacion local: 96 pruebas aprobadas, dos pruebas CUDA omitidas. CI ampliado a benchmarks en CPU en Windows y Linux; su estado remoto se consulta en los checks del pull request de esta revision.
