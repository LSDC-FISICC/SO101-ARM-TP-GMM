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

Debe ser `53f0eed` («Real robot: squeeze while holding…») o un commit posterior.

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
python -m pytest -q tests/test_gripper_command_and_overlay.py
```

Debe decir `3 passed`.

## 7. Planear y revisar la escena

Con el brazo en reposo y los cubos en la mesa, usa una foto de la cámara fija (640×480):

```bash
python scripts/infer.py --image foto_640.png --colors red yellow green
```

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

La imagen se guarda en `outputs/inference/foto_640_plan.jpg`; la detección sola, en
`outputs/inference/foto_640_detections.jpg`. Si la ventana no abre, el script intenta con matplotlib o
con el visor de imágenes del sistema; si nada funciona, abre el archivo a mano. Con `--no-show` solo
la guarda.

## 8. Prueba en seco (sin mover el robot)

```bash
python -m so101_tpgmm.robot.so101_runner --npz outputs/inference/foto_640.npz --dry-run
```

Debe terminar con `sent ... commands (dry run)`.

## 9. Ejecutar en el robot

Con el brazo en reposo, espacio libre y la mano en el interruptor:

```bash
python -m so101_tpgmm.robot.so101_runner --npz outputs/inference/foto_640.npz --port /dev/ttyACM0
```

El aviso «Relative goal position magnitude had to be clamped» aparece cuando un comando pide moverse
más de 8° de golpe. Si sale al inicio, probablemente el brazo no estaba exactamente en la pose de
reposo con que se planeó: déjalo en reposo antes de planear, o pasa su pose actual al planear con
`--start-joints` (cinco ángulos en grados).

## 10. Ajustar la fuerza de la pinza

Cambia `hold_command` y repite desde el paso 7 (planear) y el paso 9 (ejecutar):

- **No levanta el cubo:** bájalo, por ejemplo a 0. Si a 0 tampoco levanta, el problema está en la
  posición del agarre y no en la fuerza: revisa en el paso 7 si el anillo cae sobre el cubo.
- **Aprieta demasiado o el servo de la pinza se calienta:** súbelo, por ejemplo a 8.

```bash
python scripts/infer.py --image foto_640.png --colors red yellow green --set inference.gripper_command.hold_command=8
```

## 11. Recuperar cambios guardados (si hiciste `git stash`)

```bash
git stash pop
```

## Nota sobre la simulación

En MuJoCo la pinza sí apretaba con el comando 20: el modelo simulado calcula con otra geometría el
punto en que las mordazas tocan el cubo. Hay que tenerlo presente al comparar resultados de
simulación con el robot.
