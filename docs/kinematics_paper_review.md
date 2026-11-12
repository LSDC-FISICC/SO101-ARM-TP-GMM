# Revisión de la cinemática de Rivas (2026) contra el URDF del SO-101

Documento revisado: S. Rivas, *Modelado, Simulación, Teleoperación y Aprendizaje por Imitación con
ACT del Manipulador SO-101*, Universidad Galileo, LSDC (2026). Secciones II-A (DH), II-B (cinemática
inversa) y Cuadros I, II y V.

Referencia de comparación: `so101_new_calib.urdf` (TheRobotStudio/SO-ARM100), que es el modelo que
usan LeRobot y MuJoCo, con ángulos articulares en la convención de LeRobot (modo `DEGREES`).

Todo lo que sigue está verificado por tests en `tests/test_analytic_ik.py`. La versión corregida de
la IK cerrada está implementada en `so101_tpgmm/kinematics/analytic.py`.

## Veredicto

**El modelo es correcto en su estructura y coincide con el robot real.** Los puntos de la Tabla V
coinciden con el URDF hasta 0,3 mm en las cinco poses definidas, si se interpretan como el punto
sobre el eje de `wrist_roll` y el origen del marco se pone en el eje de pan. El desacople
cinemático, el brazo plano equivalente con el primer eslabón en L y el ángulo β son la forma
adecuada de resolver este brazo.

Hay **seis correcciones**: dos de convención, dos de geometría que faltan en las ecuaciones
impresas, una singularidad y una ambigüedad de notación. Con ellas la IK cerrada es exacta para el
URDF: recupera la configuración original con menos de 0,05° de error, y el residuo de ~10 µm viene
del redondeo de constantes del URDF. También es útil en la práctica como primera semilla de la
IK numérica. Con el pitch de la demostración, la IK resuelve 300 de 300 casos aleatorios con una
mediana de 8 evaluaciones (12 ms por punto). Sin la semilla cerrada, solo con semillas por azimut,
resolvía 299 de 300 en unos 14 ms.

## Parámetros obtenidos del URDF

Los valores se extraen automáticamente con `ArmGeometry.from_urdf`; el artículo no da valores
numéricos.

| Símbolo del artículo | Significado | Valor |
|---|---|---|
| l₀ + l₂ | adelanto del eje de `shoulder_lift` respecto del eje de pan | 30,4 mm |
| l₁ + l₃ | altura del eje de `shoulder_lift` sobre el origen de `base_link` | 116,6 mm |
| l₅, l₄ | brazo: componente horizontal y vertical con q₂ = 0 | 28,0 mm; 112,6 mm |
| L₁ = √(l₄² + l₅²) | longitud equivalente del brazo | 116,0 mm |
| φ = atan2(l₄, l₅) | ángulo del brazo con q₂ = 0 | 76,03° |
| l₆ | antebrazo, componente a lo largo | 134,9 mm |
| **l₆b** (nuevo) | **antebrazo, componente transversal** | **5,2 mm** (φ₂ = atan2(l₆b, l₆) = 2,21°) |
| l₇ + l₈ | eje de `wrist_flex` → punto del TCP sobre el eje de roll | 159,2 mm |
| — | `gripper_frame_link` respecto del eje de roll | 7,9 mm a lo largo del eje de cierre |
| — | origen de `base_link` respecto del eje de pan | 38,8 mm hacia atrás |

## Correcciones

### 1. Signos de q₁ y q₅ respecto de LeRobot (convención)

En el URDF, el eje de `shoulder_pan` apunta hacia abajo (−z). Un pan positivo en LeRobot gira el
brazo en sentido horario visto desde arriba, mientras que la ecuación (1) y q₁ = atan2(P_y, P_x)
suponen sentido antihorario.

- **Evidencia:** con la ecuación (1) tal como está y q = Pose A, se obtiene y = +0,0985 m. La
  propia Tabla V dice y = −0,0989 m, igual que el URDF.
- **Corrección:** q₁ = −q_pan(LeRobot) y q₅ = π − q_roll(LeRobot). Para usar la IK con el robot hay
  que aplicar esta conversión; si no, el brazo va al punto simétrico respecto del eje x.

### 2. El antebrazo también es un eslabón en L (geometría)

El vector del codo a la muñeca es (134,9; 5,2) mm, no (l₆, 0). La ecuación (1) impresa no tiene este
término. La Tabla V sí lo incluye (Z = 0,2344 m en la pose cero = 116,6 + 112,6 + 5,2 mm), así que el
código probablemente lo tiene y el artículo no.

- **Error si se omite:** hasta 5,2 mm.
- **Corrección en (1):** donde aparece l₆c₂₃ en la componente radial va l₆c₂₃ + l₆b·s₂₃, y en z,
  −l₆s₂₃ + l₆b·c₂₃.
- **Corrección en la IK:** L₂ = √(l₆² + l₆b²), y q₃ = ±arccos(·) − φ + φ₂.

### 3. Distancia radial en el desacople (convención o error tipográfico)

En II-B: x′ = x − (l₀ + l₂) − (l₇ + l₈)c_β. Debe usarse la **distancia radial**
r = √(P_x² + P_y²) (o P_x·c₁ + P_y·s₁), no x. Con q₁ ≠ 0, usar x da un resultado incorrecto. Los
casos aleatorios de la Tabla V tienen q₁ ≠ 0 y error cero, lo que sugiere que el código usa r.

### 4. β es singular en q₁ = ±90° (singularidad numérica)

β = atan2(−r₃₃, r₁₃/c₁) divide por cos q₁. En este dataset el pan llega a −98,6°, así que cruza ese
valor.

- **Corrección:** β = atan2(−r₃₃, r₁₃·c₁ + r₂₃·s₁). Es equivalente y no tiene división.
- **Test:** `test_pan_near_90_deg_is_not_singular`.

### 5. s₃ y c₃ en la ecuación (6) son del ángulo equivalente (notación)

En q₂ = atan2(−z′, r′) − atan2(L₂s₃, L₁ + L₂c₃) + atan2(l₄, l₅), y en la ley de cosenos
D² = L₁² + L₂² + 2L₁L₂c₃, el ángulo es el **relativo del brazo plano equivalente**,
θ₃ = φ₂ − φ − q₃ (signo según la rama), no q₃. Implementar literalmente sin(q₃) y cos(q₃) da
errores de **68° a 109° en q₂** en las poses A–D. Conviene escribirlo como s_θ₃ y c_θ₃.

### 6. Ramas de solución y punto final

- **Ramas:** q₁ = atan2(P_y, P_x) da solo la rama "brazo hacia adelante". Existe una segunda rama,
  q₁ + π con r → −r (brazo echado hacia atrás por encima del hombro). Junto con ± en q₃ son **hasta
  4 soluciones**, que hay que filtrar por límites articulares.
- **Punto final:** el "efector" del artículo es el punto del eje de `wrist_roll` a la profundidad
  del TCP. El `gripper_frame_link` de LeRobot está 7,9 mm fuera de ese eje, en la dirección de
  cierre de las mordazas, y la pinza está montada con un giro de 2,8° alrededor del eje de roll.
  Para la pose del TCP de LeRobot hay que restar ese desplazamiento antes del desacople:
  p_roll = p_TCP − R·(7,9; −0,2; 0) mm.
- **Errata menor:** en q₅ aparece "c1" en lugar de c₁.

## Cuadro II corregido

Convención de LeRobot/URDF para q₂, q₃ y q₄ (coincide con la del artículo). Para q₁ y q₅ se indica la
conversión. P es el punto sobre el eje de roll, expresado en el marco con origen en el eje de pan.

| Variable | Expresión |
|---|---|
| P | p_TCP − R·d_tool − (38,8; 0; 0) mm |
| q₁ (artículo) | atan2(P_y, P_x) − atan2(e, r) **y** la rama + π; LeRobot: q_pan = −q₁ |
| r | √(P_x² + P_y² − e²), con e ≈ −0,2 mm de desfase lateral |
| β | atan2(−r₃₃, r₁₃c₁ + r₂₃s₁) |
| x′, z′ | r − (l₀ + l₂) − (l₇ + l₈)c_β ; P_z − (l₁ + l₃) + (l₇ + l₈)s_β |
| c_θ₃ | (x′² + z′² − L₁² − L₂²) / (2L₁L₂), con L₂ = √(l₆² + l₆b²) |
| θ₃ | ±arccos(c_θ₃) |
| q₂ | φ − atan2(z′, x′) + atan2(L₂ sin θ₃, L₁ + L₂ cos θ₃) |
| q₃ | φ₂ − φ − θ₃ |
| q₄ | β − q₂ − q₃ |
| q₅ | ángulo de R₀ᵀR alrededor del eje de roll (Rodrigues); LeRobot: q_roll = π − q₅ |

## Sobre la validación de la Sección III-A

El "error nulo" de la Tabla V compara FK→IK→FK con el mismo modelo, así que demuestra consistencia
algebraica, no que el modelo represente al robot. Un error de convención presente en ambas
direcciones (por ejemplo el signo de q₁) no aparece en esa prueba.

Recomendación: validar contra una referencia independiente, como la FK del URDF o MuJoCo (lo que se
hizo aquí) o una medición física del TCP en dos o tres poses. Esa última también ayudaría a
resolver la diferencia de escala de 4,7 % que encontramos entre la FK y la cuadrícula de cinta
(`docs/camera_calibration.md`).
