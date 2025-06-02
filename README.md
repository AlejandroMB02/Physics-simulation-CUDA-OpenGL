# 💥🌌 PARTICLE-SIM: Simulación de Partículas con CUDA + OpenGL

![CUDA](https://img.shields.io/badge/CUDA-GPU%20Computing-76B900)
![OpenGL](https://img.shields.io/badge/OpenGL-Rendering-5586A4)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue)

**PARTICLE-SIM** es una simulación de partículas acelerada por GPU, diseñada para visualizar físicas básicas de forma dinámica en tiempo real utilizando CUDA y OpenGL. Con este proyecto exploramos la interoperabilidad entre computación paralela y gráficos 3D, y sirve como entorno de pruebas personal para simulaciones físicas.

![ParticleSim](/img/gif_app.gif)

## ✨ Características Principales

- Simulación de partículas con movimiento dinámico
- Cálculos de física en GPU mediante CUDA
- Renderizado en tiempo real con OpenGL
- Interoperabilidad CUDA ↔ OpenGL vía VBOs

## 🛠️ Tecnologías Utilizadas

- **CUDA** – Cómputo paralelo en GPU
- **OpenGL** – Renderizado de gráficos
- **GLFW** – Gestión de ventanas y entrada
- **GLAD** – Cargador de funciones de OpenGL
- **CMake** – Sistema de construcción multiplataforma

## 🚀 Cómo Ejecutar

### 1. Requisitos

- GPU NVIDIA compatible con CUDA
- CUDA Toolkit instalado
- OpenGL y GLFW disponibles en el sistema
- CMake
- Compilador con soporte C++17

### 2. Clonar y Compilar

```bash
git clone https://github.com/AlejandroMB02/Physics-simulation-CUDA-OpenGL
mkdir build && cd build
cmake ..
make
./ParticleSim
