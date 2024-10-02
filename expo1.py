import streamlit as st

def main():
    # Título
    st.title("Exposición: Creación de GNU/Linux y su impacto actual")

    # Sección 1: Origen de GNU/Linux
    st.header("1. Creación de GNU/Linux")
    st.write("""
    GNU/Linux es un sistema operativo libre y de código abierto que surgió de la combinación del proyecto GNU, iniciado por Richard Stallman en 1983, y el kernel Linux, desarrollado por Linus Torvalds en 1991. El objetivo principal del proyecto GNU era crear un sistema operativo completamente libre, y la llegada del kernel Linux fue crucial para completar este objetivo. La comunidad de desarrolladores alrededor de ambos proyectos creció exponencialmente, llevando a la creación de diversas distribuciones de GNU/Linux.
    """)

    # Sección 2: Eventos destacados en la historia de GNU/Linux
    st.header("2. Eventos destacados")
    st.write("""
    Algunos de los eventos más importantes en la evolución de GNU/Linux incluyen:
    
    - **1991**: Linus Torvalds publica la primera versión del kernel Linux.
    - **1993**: Nacen las primeras distribuciones populares como Slackware y Debian.
    - **2004**: Se lanza Ubuntu, que democratiza el uso de GNU/Linux para usuarios no técnicos.
    - **2012**: GNU/Linux se convierte en el sistema dominante en servidores y plataformas en la nube.
    """)

    # Sección 3: Intervención en tecnologías actuales
    st.header("3. Intervención en tecnologías actuales")
    st.write("""
    Hoy en día, GNU/Linux juega un papel central en muchas áreas tecnológicas:
    
    - **Servidores**: Más del 90% de los servidores en la web funcionan con distribuciones de Linux debido a su estabilidad y seguridad.
    - **Supercomputación**: Los 500 superordenadores más potentes del mundo usan Linux.
    - **Desarrollo de software**: Herramientas como Docker y Kubernetes, que son esenciales en la infraestructura de software moderna, están construidas sobre GNU/Linux.
    - **IoT y dispositivos embebidos**: Muchas tecnologías IoT utilizan Linux como sistema operativo base debido a su flexibilidad y capacidad de personalización.
    """)

    # Conclusión
    st.header("Conclusión")
    st.write("""
    GNU/Linux ha sido fundamental en la transformación tecnológica que hemos vivido en las últimas décadas. Su modelo de código abierto ha fomentado la innovación y la colaboración global, siendo ahora una parte integral de sistemas críticos en todas partes del mundo, desde servidores hasta supercomputadoras y dispositivos embebidos.
    """)

if __name__ == "__main__":
    main()
