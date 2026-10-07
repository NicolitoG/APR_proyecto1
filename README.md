# Procesar tareas en red satelital

## Intro 
El objetivo es decidir qué satélite estará disponible para procesar tareas originadas por usuarios en tierra. Se proveerá el ambiente en código en Python usando Gymnasium y se disponibilizará la data del ambiente para cada paso $\delta t$.

## Ambiente
- Se explica el ambiente con un caso de prueba utilizando solo dos satélites y un usuario.
- El usuario en tierra genera tareas aleatoriamente. No genera una tarea adicional mientras no haya completado o fallado la tarea actual. La tarea se completa exitosamente si logra cargarla a un satélite en línea de vista activo para procesar. En caso de que exista más de un satélite que cumpla los requisitos, el usuario enviará la tarea al más cercano. Una tarea falla cuando han pasado $R_T$ segundos.
- Los satélites orbitan la Tierra en trayectorias tipo LEO. Tienen dos estados: 1) estado activo, en el que procesan tareas, y 2) estado pasivo, en el que no pueden procesar. Cuando los satélites ven a los usuarios, el ambiente provee varios datos adicionales que pueden ser utilizados como estado (input) del modelo de aprendizaje.

## Restricciones
- Los satélites poseen una batería que se carga y descarga durante la órbita de acuerdo con sus paneles solares. Cuando la Tierra bloquea al Sol, el satélite se descarga (noche), mientras que cuando tiene línea directa al Sol se carga (día). Esto pasa en ciclos de órbitas de aproximadamente 90 minutos.
- Adicionalmente, la batería se descarga más rápidamente cuando el satélite está en estado activo y sufrirá una descarga adicional por procesar una tarea.
- Cuando la batería llega a cierta capacidad límite, el satélite pierde la facultad de procesar tareas mientras no se cargue nuevamente. 

## Recompensa, estado y política

- La recompensa suma las tareas procesadas exitosamente y resta las fallidas.
- El estado puede ser construido con diversas variables del ambiente. Entre otras, se pueden ocupar para cada instante t y satélite s:
    - Número de usuarios en línea de vista u_s
    - Vector de identidad de usuario U_s (u_s x 1)
    - Vector de largo de tareas pendientes por usuario T_s (u_s x 1)
    - Vector de tiempo restante para que una tarea falle R_s (u_s x 1) 
    - Vector de distancia a usuarios en línea de vista D_s (u_s x 1)
    - Posición del satélite P_s = (x_s, y_s, z_s) y L_s = (lat_s, lon_s, h_s)
    - Carga de la batería c_s y estado e_s (cargando o descargando)
- La política de decisión Y es un vector (S x 1) con la decisión de activar (1) o desactivar (0) el procesamiento en los satélites.

## Entrenamiento

- El entrenamiento de los modelos se basa en diferentes episodios de tiempo T.
- La suma de todos los pasos t hace todo el episodio.
- Durante un episodio se mantiene fija la constelación (el número de satélites S), pero pueden aparecer o desaparecer usuarios y tareas de diferentes largos.
- En cada instante t, dadas las variables del ambiente, el modelo debe elegir una política Y que se aplicará a los satélites y se evaluará con recompensa r.
