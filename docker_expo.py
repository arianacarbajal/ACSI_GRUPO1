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
    5. Referencias
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

def mostrar_docker_vs_vm():
    st.header("Docker vs Máquinas Virtuales")
    st.markdown("""
    **Docker** utiliza contenedores ligeros que comparten el kernel del sistema operativo host, lo que lo hace más eficiente 
    que las **Máquinas Virtuales (VMs)**, que requieren un sistema operativo completo y recursos dedicados.

    **Comparativa Docker vs VMs:**
    - **Docker**: Ligero, rápido, comparte recursos del sistema host.
    - **VMs**: Necesitan más recursos, cada VM tiene su propio SO, mayor aislamiento.
    """)

def mostrar_instalacion():
    st.header("Instalación de Docker en Ubuntu 20.04.6 LTS")
    st.markdown("""
    Para instalar Docker en Ubuntu, sigue los siguientes pasos:

    1. Actualizar el sistema:
       ```bash
       sudo apt-get update
       ```

    2. Instalar paquetes necesarios para manejar repositorios a través de HTTPS:
       ```bash
       sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
       ```

    3. Descargar la llave GPG oficial de Docker:
       ```bash
       curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
       ```

    4. Verificar la llave:
       ```bash
       sudo apt-key fingerprint 0EBFCD88
       ```

    5. Agregar el repositorio de Docker para la arquitectura x86_64/amd64:
       ```bash
       sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
       ```

    6. Instalar Docker:
       ```bash
       sudo apt-get update
       sudo apt-get install docker-ce docker-ce-cli containerd.io
       ```

    7. Verificar la instalación de Docker:
       ```bash
       sudo docker version
       ```

    8. Añadir el usuario actual al grupo `docker` para evitar usar `sudo` cada vez:
       ```bash
       sudo usermod -aG docker [tu_usuario]
       ```

    9. Cierra la sesión y vuelve a iniciar sesión. Luego verifica la instalación:
       ```bash
       docker version
       ```
    """)

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

def mostrar_referencias():
    st.header("Referencias y Recursos Adicionales")
    st.markdown("""
    **Sitios recomendados:**
    - [Documentación oficial de Docker](https://docs.docker.com)
    - [Docker Hub](https://hub.docker.com)
    - [Guía rápida de Docker en GitHub](https://github.com/docker)
    """)

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
elif seleccion == "Referencias":
    mostrar_referencias()

# Añadir footer interactivo
st.sidebar.markdown("**Duración total estimada:** 10 minutos.")
st.sidebar.markdown("[GitHub Repo](https://github.com/docker/docker) | [Más Info](https://www.docker.com)")
