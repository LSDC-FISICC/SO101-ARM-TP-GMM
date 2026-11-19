# TP-GMM en el SO-101 real: pasos

Ejecutar el TP-GMM en el robot desde la laptop conectada al brazo y a la cámara fija: actualizar el
código, planear a partir de una foto, revisar la escena y mandar la trayectoria al robot.

## Antes de empezar: por qué cambió el comando de la pinza

El TP-GMM aprendió la posición **medida** de la pinza del seguidor. Mientras el operador sostenía un
cubo, esa posición era ~20, porque el cubo de 4 cm frena las mordazas. El operador, en cambio,
comandaba con el líder la pinza casi cerrada (~0,8; mediana en los 400 ciclos de dataset3). Si se
envía 20, las mordazas llegan justo al cubo sin error de posición, el servo no aprieta y el cubo no
sube.

Desde el commit `53f0eed`, el comando que se envía mientras el plan sostiene el cubo es
`hold_command` (2 por defecto):

| Valor de la pinza en el plan | Comando enviado |
|---|---|
| ≥ 26 (abierta o abriéndose) | el del plan |
| entre 20 y 26 | interpolado, sin saltos |
| ≤ 20 (sosteniendo) | `hold_command` |

Configuración: `inference.gripper_command` en `configs/default.yaml`. La trayectoria no cambia.

## Antes de empezar: la vista de la cámara

En la prueba del 2026-11 los cubos rojo y amarillo chocaron con la punta de un dedo y el verde se
entregó. La causa fue la **posición**, no la fuerza. La foto usada (`foto_640.png`) no tenía la vista
de la calibración: la cuadrícula de cinta se veía ~11 % más chica y desplazada (otra resolución
redimensionada, otro zoom o la cámara movida). La homografía píxel → base solo vale para la vista de
dataset3, así que los agarres quedaron corridos 12 mm (verde), 13 mm (rojo) y 26 mm (amarillo). Con
la pinza abierta, el margen a cada lado del cubo es de ~12 mm.

Desde este cambio:

- `scripts/capture_top.py` toma la foto a 640×480 en el modo de la grabación y no la guarda si la
  cámara entrega otro tamaño.
- `scripts/infer.py` alinea la cinta de la foto con la vista de calibración
  (`configs/reference_top.jpg`) y corrige la homografía (`--register on`, por defecto). La cinta y el
  robot están fijos a la mesa, así que la cinta sirve de patrón de calibración en cada foto. No
  corrige si el robot o la cinta se movieron uno respecto del otro.

## 1. Entrar al repositorio y activar el entorno

```bash
cd ~/so101_arm/SO101-ARM-TP-GMM
```

```bash
source .venv/bin/activate
```

## 2. Ver la rama actual y si hay cambios sin guardar

```bash
git branch --show-current
```

```bash
git status
```

Si `git status` lista archivos modificados (`modified:`) que quieres conservar, guárdalos aparte antes
de cambiar de rama:

```bash
git stash
```

Se recuperan al final con `git stash pop`. Los archivos de `outputs/` (modelos, calibración, fotos
procesadas) no están versionados y no cambian al cambiar de rama.

## 3. Traer los cambios y pasar a la rama de trabajo

```bash
git fetch origin
```

```bash
git checkout claude/stoic-wright-rcmidc
```

Si dice que la rama no existe localmente, créala siguiendo a la remota:

```bash
git checkout -b claude/stoic-wright-rcmidc origin/claude/stoic-wright-rcmidc
```

Luego actualiza:

```bash
git pull
```

## 4. Verificar la versión

```bash
git log --oneline -1
```

Debe ser `a8e8c01` («Real robot: register the camera view on the taped grid…») o uno posterior (incluye
también el apriete de la pinza de `53f0eed`).

```bash
grep -n "gripper_command" configs/default.yaml
```

Debe mostrar `gripper_command: {contact_level: 20.0, release_level: 26.0, hold_command: 2.0}`.

## 5. Confirmar que están los modelos y la calibración

```bash
ls outputs/models/ outputs/vision_calibration.yaml
```

Deben aparecer `segmented.json` (y los demás modelos) y el archivo de calibración.

## 6. Prueba rápida del código nuevo

```bash
python -m pytest -q tests/test_gripper_command_and_overlay.py tests/test_view_registration.py
```

Debe decir `4 passed`.

## 7. Tomar la foto

Con el brazo en reposo y los cubos en la mesa, captura con la cámara fija de arriba (índice 3 en la
grabación; si no es ese, prueba 2, 4… o `/dev/videoN`):

```bash
python scripts/capture_top.py --camera 3 --out foto.png
```

No uses una foto de otro programa redimensionada: otro modo de captura puede cambiar el campo de
visión. El script imprime una línea así:

```
view vs calibration: scale 0.998, alignment ECC 0.99, uncorrected grasp error median 1.3 mm, max 2.6 mm
```

- `ECC` ≥ 0,9: la cinta se alineó bien. Por debajo de 0,7 no se corrige: revisa que sea la cámara de
  arriba y que la cuadrícula se vea completa (sin manos ni objetos encima).
- `uncorrected grasp error`: cuánto se corrían los agarres sin corregir. Con la foto de la prueba
  anterior daba mediana 22 mm, máx. 39 mm. Si pasa de ~5 mm, la cámara se movió desde la calibración;
  el paso 8 lo corrige, pero conviene devolverla a su lugar.
- `foto_vs_reference.jpg`: la vista de calibración deformada sobre la foto. La cinta debe verse
  sencilla, no doble.

Con `--preview` se ve la imagen en vivo antes de guardar (espacio guarda, q sale).

## 8. Planear y revisar la escena

```bash
python scripts/infer.py --image foto.png --colors red yellow green
```

Imprime de nuevo la línea `view vs calibration` y, si la vista difiere, `(corrected)`.

Se abre una ventana con la escena (cualquier tecla la cierra):

- la detección de cada cubo, con su número de orden y `OK` (planeado) o `SKIP` (omitido);
- el recorrido planeado del TCP proyectado sobre la mesa, en el color del cubo: trazo grueso hasta el
  agarre y delgado en el transporte;
- un **rombo magenta** donde está la pinza ahora según la cinemática del robot (FK), proyectado con
  la calibración de la cámara;
- en el agarre, un **punto** donde quedará el TCP y un **anillo con cruz** donde quedará el centro del
  cubo según la calibración.

Dos comprobaciones distintas:

- **El rombo magenta debe caer entre las puntas de la pinza que se ve en la foto.** Es la única marca
  que compara la cinemática del robot con la cámara: si cae lejos, la cámara se movió desde la
  calibración y todas las posiciones están desplazadas (aunque los anillos caigan sobre los cubos).
  Supone que el brazo estaba en la pose de reposo al tomar la foto, o la que se pase con
  `--start-joints`.
- **El anillo debe caer sobre el centro del cubo detectado.** Comprueba que el plan apunta al cubo. No
  comprueba la cámara: detección y anillo pasan por la misma homografía, así que una cámara movida los
  desplaza juntos. Un anillo en el borde del cubo anticipa un agarre por el borde.

La imagen se guarda en `outputs/inference/foto_plan.jpg`; la detección sola, en
`outputs/inference/foto_detections.jpg`. Si la ventana no abre, el script intenta con matplotlib o
con el visor de imágenes del sistema; si nada funciona, abre el archivo a mano. Con `--no-show` solo
la guarda.

## 9. Prueba en seco (sin mover el robot)

```bash
python -m so101_tpgmm.robot.so101_runner --npz outputs/inference/foto.npz --dry-run
```

Debe terminar con `sent ... commands (dry run)`.

## 10. Ejecutar en el robot

Con el brazo en reposo, espacio libre y la mano en el interruptor:

```bash
python -m so101_tpgmm.robot.so101_runner --npz outputs/inference/foto.npz --port /dev/ttyACM0
```

El aviso «Relative goal position magnitude had to be clamped» aparece cuando un comando pide moverse
más de 8° de golpe. Si sale al inicio, probablemente el brazo no estaba exactamente en la pose de
reposo con que se planeó: déjalo en reposo antes de planear, o pasa su pose actual al planear con
`--start-joints` (cinco ángulos en grados).

## 11. Ajustar la fuerza de la pinza

Cambia `hold_command` y repite desde el paso 8 (planear) y el paso 10 (ejecutar):

- **No levanta el cubo:** bájalo, por ejemplo a 0. Si a 0 tampoco levanta, el problema está en la
  posición del agarre y no en la fuerza: revisa en el paso 8 si el anillo cae sobre el cubo y el rombo sobre la pinza.
- **Aprieta demasiado o el servo de la pinza se calienta:** súbelo, por ejemplo a 8.

```bash
python scripts/infer.py --image foto.png --colors red yellow green --set inference.gripper_command.hold_command=8
```

## 12. Recuperar cambios guardados (si hiciste `git stash`)

```bash
git stash pop
```

## Nota sobre la simulación

En MuJoCo la pinza sí apretaba con el comando 20: el modelo simulado calcula con otra geometría el
punto en que las mordazas tocan el cubo. Hay que tenerlo presente al comparar resultados de
simulación con el robot.
