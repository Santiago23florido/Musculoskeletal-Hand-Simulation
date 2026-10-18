# Piloto de deformación geométrica de MUAP

**Estado: piloto de entrenamiento/validación, sin predicciones externas.**
Tras descartar en validación el reajuste libre de la CNN, se probó una
deformación de la plantilla inicial con 240 parámetros compartidos.
La salida conserva la identidad en q=0, cero entrada produce cero salida
y escalar la amplitud de la entrada escala la salida de la misma forma.
No hay un patrón aditivo común ni parámetros por identificador.

El operador desplaza/estira tiempo y desplaza, escala y rota la cuadrícula
espacial. Sus ocho parámetros son funciones acotadas de q, q², q³ y diez
momentos de energía de la plantilla de calibración. Usa interpolación
trilineal determinista y un límite de corrección relativo a la norma propia.
Las pruebas verifican identidad, homogeneidad de amplitud, permutación de
unidades y gradiente CUDA finito.

Con la pérdida proyectada dominante, la selección por riesgo de fuente
conserva la identidad. Un piloto separado con riesgo de fuente dominante
reduce ese riesgo de validación de 0,7151 a 0,6498 en seis épocas, mientras
empeoran el error de forma y el proyectado. Sus pesos se descartan; no se
midió RoA externo ni se afirma que esa reducción recupere descargas.

Los protocolos, fuentes, historiales y pilotos se conservan en
`outputs/stage_b_geometry/`. La corrección que pasa a evaluación de disparos
está en [spike_likelihood](../spike_likelihood/README.md), con etiquetas de
eventos y selección de umbral por Jaccard de validación. Sus resultados están
en el [resumen publicado](../../docs/benchmarks/spike_likelihood_summary.json).
