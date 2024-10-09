import streamlit as st

# Configuración básica de la página
st.set_page_config(page_title="Introducción a Docker", layout="wide")

# Título de la presentación
st.title("Presentación Interactiva: Introducción a Docker")

# Sidebar para navegar por los temas
st.sidebar.title("Navegación")
opciones = [
    "Índice", 
    "¿Qué es Docker?", 
    "Docker vs Máquinas Virtuales", 
    "Instalación de Docker", 
    "Conceptos Básicos", 
    "Imágenes y Contenedores", 
    "Comandos Esenciales de Docker",
    "Referencias"
]
seleccion = st.sidebar.radio("Ir a la sección:", opciones)

# Función para cada sección
def mostrar_indice():
    st.header("Índice")
    st.markdown("""
    1. ¿Qué es Docker?
    2. Docker vs Máquinas Virtuales
    3. Instalación de Docker
    4. Conceptos Básicos y Comandos de Docker
    5. Imágenes y Contenedores
    6. Comandos Esenciales
    7. Referencias
    """)

def mostrar_que_es_docker():
    st.header("¿Qué es Docker?")
    st.markdown("""
    Docker es una plataforma de contenedorización que permite a los desarrolladores empaquetar sus aplicaciones y 
    todas sus dependencias en un contenedor. Esto asegura que la aplicación se ejecutará de la misma manera en 
    cualquier entorno, ya sea en desarrollo, pruebas o producción.

    **Ventajas de Docker:**
    - Aislamiento de aplicaciones.
    - Despliegue rápido y eficiente.
    - Escalabilidad mejorada.
    """)
    st.image("https://www.docker.com/sites/default/files/d8/2019-07/Moby-run.png", caption="Contenedores Docker en acción")

def mostrar_docker_vs_vm():
    st.header("Docker vs Máquinas Virtuales")
    st.markdown("""
    **Docker** utiliza contenedores ligeros que comparten el kernel del sistema operativo host, lo que lo hace más eficiente 
    que las **Máquinas Virtuales (VMs)**, que requieren un sistema operativo completo y recursos dedicados.

    **Comparativa Docker vs VMs:**
    - **Docker**: Ligero, rápido, comparte recursos del sistema host.
    - **VMs**: Necesitan más recursos, cada VM tiene su propio SO, mayor aislamiento.

    """)
    st.image("https://phoenixnap.com/kb/wp-content/uploads/2021/11/docker-vm-comparison.png", caption="Docker vs Máquinas Virtuales")

def mostrar_instalacion():
    st.header("Instalación de Docker")
    st.markdown("""
    **Pasos para instalar Docker en diferentes sistemas operativos:**

    - **Linux**:  
      ```bash
      sudo apt-get update
      sudo apt-get install docker-ce
      ```
    - **Windows** y **macOS**:  
      Descargar e instalar [Docker Desktop](https://www.docker.com/products/docker-desktop).

    Tras la instalación, puedes verificar que Docker está instalado correctamente ejecutando:
    ```bash
    docker --version
    ```
    """)
    st.image("https://miro.medium.com/max/3000/1*k7Z4G4iq-HbhPs4lzh99Ew.png", caption="Docker Desktop en Windows")

def mostrar_conceptos_basicos():
    st.header("Conceptos Básicos de Docker")
    st.markdown("""
    **Imágenes**: Son plantillas de solo lectura que contienen el sistema operativo y las aplicaciones necesarias.

    **Contenedores**: Son instancias ejecutables de las imágenes. Representan la ejecución de aplicaciones.

    **Docker Hub**: Es un repositorio en línea donde puedes encontrar miles de imágenes predefinidas listas para usar.

    **Comandos clave**:
    - `docker pull`: Descargar una imagen desde Docker Hub.
    - `docker run`: Ejecutar un contenedor a partir de una imagen.
    - `docker ps`: Ver los contenedores en ejecución.
    - `docker stop`: Detener un contenedor en ejecución.
    """)
    st.image("https://docs.docker.com/get-started/images/docker-swarm.png", caption="Arquitectura de Docker")

def mostrar_imagenes_contenedores():
    st.header("Imágenes y Contenedores")
    st.markdown("""
    **Imágenes**:
    Son plantillas de solo lectura que contienen todo lo necesario para ejecutar una aplicación: código, dependencias, 
    librerías, etc. Puedes crear una imagen personalizada o usar imágenes existentes desde Docker Hub.

    **Contenedores**:
    Los contenedores son instancias en ejecución de imágenes. Se ejecutan en un entorno aislado y contienen solo lo 
    necesario para la aplicación. Puedes ejecutar, pausar, detener y eliminar contenedores con facilidad.

    **Gestión de Imágenes**:
    - Buscar imágenes en Docker Hub:
      ```bash
      docker search nombre_imagen
      ```
    - Descargar imágenes desde Docker Hub:
      ```bash
      docker pull nombre_imagen
      ```

    **Gestión de Contenedores**:
    - Crear y ejecutar contenedores:
      ```bash
      docker run -d -p 80:80 nombre_imagen
      ```
    - Inspeccionar contenedores:
      ```bash
      docker inspect nombre_contenedor
      ```
    """)
    st.image("https://docs.docker.com/get-started/images/inspect-container.png", caption="Inspección de Contenedores en Docker")

def mostrar_comandos_esenciales():
    st.header("Comandos Esenciales de Docker")
    st.markdown("""
    Aquí están algunos comandos clave para trabajar con Docker de manera eficiente:

    - `docker images`: Ver todas las imágenes descargadas localmente.
    - `docker ps`: Ver los contenedores en ejecución.
    - `docker stop [ID]`: Detener un contenedor en ejecución.
    - `docker rm [ID]`: Eliminar un contenedor detenido.
    - `docker rmi [nombre_imagen]`: Eliminar una imagen.
    - `docker exec -it [nombre_contenedor] bash`: Acceder al terminal dentro de un contenedor.
    """)
    st.image("https://phoenixnap.com/kb/wp-content/uploads/2021/11/docker-stop-command.png", caption="Uso de comandos esenciales en Docker")

def mostrar_referencias():
    st.header("Referencias y Recursos Adicionales")
    st.markdown("""
    **Sitios recomendados:**
    - [Documentación oficial de Docker](https://docs.docker.com)
    - [Docker Hub](https://hub.docker.com)
    - [Guía rápida de Docker en GitHub](https://github.com/docker)

    **Cursos recomendados**:
    - [Docker for Developers](https://www.coursera.org/learn/docker-for-developers)
    - [Pluralsight - Docker Deep Dive](https://www.pluralsight.com/courses/docker-deep-dive)

    **Libros recomendados**:
    - *Docker: Up & Running* de O'Reilly Media
    - *The Docker Book* de James Turnbull
    """)
    st.image("https://miro.medium.com/max/1200/1*IEY9hbIcBOYTyue40bB_xg.png", caption="Logo Docker")

# Mostrar la sección seleccionada en la barra lateral
if seleccion == "Índice":
    mostrar_indice()
elif seleccion == "¿Qué es Docker?":
    mostrar_que_es_docker()
elif seleccion == "Docker vs Máquinas Virtuales":
    mostrar_docker_vs_vm()
elif seleccion == "Instalación de Docker":
    mostrar_instalacion()
elif seleccion == "Conceptos Básicos":
    mostrar_conceptos_basicos()
elif seleccion == "Imágenes y Contenedores":
    mostrar_imagenes_contenedores()
elif seleccion == "Comandos Esenciales de Docker":
    mostrar_comandos_esenciales()
elif seleccion == "Referencias":
    mostrar_referencias()

st.sidebar.markdown("[GitHub Repo](https://github.com/docker/docker) | [Más Info](https://www.docker.com)")
