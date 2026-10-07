# Procesar Tareas en Red Satelital

## Intro 
El objetivo es decidir que satelite estará disponible para procesar tareas originadas por usuarios en tierra. Se proveera el ambiente en codigo en python usando gymnasium y se disponibilizara la data de ambiente para cada paso \deltat 

## Ambiente
- Se explica el ambiente con un caso de prueba utilizando solo dos satelites y un usuario.
- El usuario en tierra genera tareas de tipo $T$ aleatoriamiente. No genera una tarea adicional mientras no haya completado o fallado la tarea actual. La tarea se completa exitosamente si logra cargarla a un satelite en linea vista activo para procesar. En caso de que exista mas de un satelite que cumpla los requisitos, el usuario enviará la tarea al mas cercano. Una tarea falla cuando han pasado $R_T$ segundos.
- Los satelites orbitan la tierra en trayectorias tipo  LEO. Tienen dos estados: 1) Estado Activo, donde procesan tareas y 2) Estado Pasivo no pueden procesar. Cuando los satelites ven a los usuarios el ambiente provee varios datos adicionales que pueden ser utilizados como Estado (Input) del modelo de aprendizaje.

## Restricciones
- Los satelites poseen una bateria que se carga y descarga durante la orbita de acuerdo a sus paneles solares. Cuando la tierra bloquea al sol el satelite se descarga (noche) mientras que cuando tiene linea directa al sol se carga (día) esto pasa en ciclos de orbitas de aproximadamente 90 minutos.
- Adicionalmente la bateria se descarga mas rapidamente cuando el satelite esta en Estado Activo y sufrirá una descarga adicional por procesar una tarea.
- Cuando la batería llega a cierta capacidad limite el satelite pierde la facultad de procesar tareas mientras no se cargue nuevamente. 

## Recompensa, Estado y Politica

- La recompensa suma las tareas procesadas exitosamente y resta las fallidas.
- El Estado puede ser contruido con diversas variables de ambiente. Entre otros se pueden ocupar para cada instante t y satelite s:
    - Numero de usuarios en linea vista u_s
    - Vector de indentidad de usuario U_s (u_s x 1)
    - Vector de largo de tareas pendiente por usuario T_s (u_s x 1)
    - Vector de tiempo restante para que una tarea falle R_s (u_s x 1) 
    - Vector Distancia a usuarios en linea vista D_s (u_s x 1)
    - Posicion del satelite P_s = (x_s, y_s, z_s) y L_s = (lat_s, lon_s, h_s)
    - Carga de la bateria c_s y estado e_s (cargando o descargando)
- La politica de desición Y es un vector (S x 1) con la desición de Activar (1) o Desactivar (0) los satelites.

## Entrenamiento

- El entremaniemto de los modelos se base en diferentes episodios de tiempo T.
- La suma de todos los pasos t hacen todo el episodio.
- Durante un episodio se mantiene fija la constelación (el numero de satelites S) pero pueden aparecer o desaparecer usuarios y tareas de diferentes largos.
- A cada instante t las variables de ambiente, el modelo debe elegir una politica Y que se aplicará a los satelites y se evaluará con recompensa r.
