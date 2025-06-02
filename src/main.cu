#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

#define BLOCKSIZE 256

struct Particle {
    float x, y;
    float vx, vy;
};

std::vector<Particle> particles; // Contiene todas las partículas de la simulación

// Variables para FPS
double lastTime = glfwGetTime();
int frameCount = 0;
double fps = 0.0;using namespace std;
double fpsAlpha = 0.1; // Factor de suavizado (0.1 = 10% de nuevo valor, 90% de valor anterior)
/////////////////////

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

const char* vertexShaderSource = R"glsl(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        gl_PointSize = 10.0; // partículas más grandes
    }
)glsl";

const char* fragmentShaderSource = R"glsl(
    #version 330 core
    out vec4 FragColor;
    void main() {
        FragColor = vec4(1.0, 0.3, 0.3, 1.0); // rojo claro
    }
)glsl";

void initParticles(int num) {
    particles.clear();
    srand(static_cast<unsigned>(time(0))); // Semilla de rand
    for (int i = 0; i < num; ++i) {
        float x = ((float)rand() / RAND_MAX) * 1.8f - 0.9f;
        float y = ((float)rand() / RAND_MAX) * 1.8f - 0.9f;
        float vx = ((float)rand() / RAND_MAX) * 0.01f - 0.005f;
        float vy = ((float)rand() / RAND_MAX) * 0.01f - 0.005f;
        particles.push_back({x, y, vx, vy});
    }
}

//*************************************************
// Function for checking CUDA runtime API results
//*************************************************
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

//*************************************************
// KERNELS 
// ************************************************
__global__ void warm_up_gpu(){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float i,j=1.0,k=2.0;
    i = j+k; 
    j+=i+float(tid);
  }

  __global__ void updateParticlesKernel(Particle* particles, int numParticles, float gravity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numParticles) {
        // Aplicar gravedad
        particles[idx].vy += gravity;
        
        // Actualizar posición
        particles[idx].x += particles[idx].vx;
        particles[idx].y += particles[idx].vy;
        
        // Rebote con bordes en X
        if (particles[idx].x < -1.0f || particles[idx].x > 1.0f) {
            particles[idx].x = (fabs(particles[idx].x)/particles[idx].x) + 
                              ((fabs(particles[idx].x)/particles[idx].x) - particles[idx].x);
            particles[idx].vx *= -1;
        }
        
        // Rebote con bordes en Y
        if (particles[idx].y < -1.0f || particles[idx].y > 1.0f) {
            particles[idx].y = (fabs(particles[idx].y)/particles[idx].y) + 
                              ((fabs(particles[idx].y)/particles[idx].y) - particles[idx].y);
            particles[idx].vy *= -1;
        }
    }
}

__global__ void collideParticlesKernel(Particle* particles, int numParticles, float minDist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= numParticles) return;
    
    // Cada hilo procesa una partícula i
    float ix = particles[i].x;
    float iy = particles[i].y;
    
    // Comparar con partículas j > i para evitar duplicados
    for (int j = i + 1; j < numParticles; j++) {
        float dx = particles[j].x - ix;
        float dy = particles[j].y - iy;
        float dist2 = dx*dx + dy*dy;
        
        if (dist2 < minDist*minDist) {
            // Intercambio de velocidades atómico
            float temp_vx = atomicExch(&particles[i].vx, particles[j].vx);
            atomicExch(&particles[j].vx, temp_vx);
            
            float temp_vy = atomicExch(&particles[i].vy, particles[j].vy);
            atomicExch(&particles[j].vy, temp_vy);
            
            // Romper el bucle después de una colisión para evitar múltiples intercambios
            break;
        }
    }
}

/////////////////////////////////////////////////
// Funciones para llamar a los kernels
/////////////////////////////////////////////////
void updateParticlesCUDA(std::vector<Particle>& particles, float gravity) {
    int numParticles = particles.size();
    
    if (numParticles == 0) return;

    // 1. Reservar memoria en GPU
    Particle* d_particles;
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));
    
    // 2. Copiar datos de CPU a GPU
    cudaMemcpy(d_particles, particles.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    
    // 3. Configurar ejecución del kernel
    int blocksPerGrid = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
    
    // 4. Lanzar kernel
    updateParticlesKernel<<<blocksPerGrid, BLOCKSIZE>>>(d_particles, numParticles, gravity);
    
    // 5. Copiar resultados de vuelta a CPU
    cudaMemcpy(particles.data(), d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
    
    // 6. Liberar memoria de GPU
    cudaFree(d_particles);
    
    // Verificar errores (opcional pero recomendado)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error en CUDA: %s\n", cudaGetErrorString(err));
    }
}

void collideParticlesCUDA(std::vector<Particle>& particles, float minDist) {
    int numParticles = particles.size();
    if (numParticles < 2) return;

    Particle* d_particles;
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));
    cudaMemcpy(d_particles, particles.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    // Configurar bloques e hilos
    int blocks = (numParticles + BLOCKSIZE - 1) / BLOCKSIZE;
    
    // Lanzar kernel
    collideParticlesKernel<<<blocks, BLOCKSIZE>>>(d_particles, numParticles, minDist);
    
    // Copiar resultados de vuelta
    cudaMemcpy(particles.data(), d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaFree(d_particles);
    
    // Verificar errores
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error en CUDA: %s\n", cudaGetErrorString(err));
    }
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Partículas con Gravedad y Colisión", nullptr, nullptr);
    if (!window) {
        std::cerr << "Error creando ventana\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //glfwSwapInterval(0); //Quita la sincronización vertical FPS 100%
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Error cargando GLAD\n";
        return -1;
    }

    glEnable(GL_PROGRAM_POINT_SIZE);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    initParticles(7000);

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(Particle), particles.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    const float gravity = -0.0002f;
    const float minDist = 0.02f; //0.02f

    ////////////////////////////////////
    //Get GPU information
    int devID=0;
    cudaDeviceProp props;
    
    cout<<"Using Device "<<devID<<endl;
    cout<<"....................................................."<<endl<<endl;
    checkCuda(cudaGetDeviceProperties(&props, devID));
    cout<<"****************************************************************************************"<<endl;
    cout<<"Using Device "<< devID<<": "<<props.name<<"  with CUDA Compute Capability "<<props.major<<"."<<props.minor<<endl;
    cout<<"****************************************************************************************"<<endl<<endl;
    checkCuda(cudaSetDevice(devID)); 

    cout<<"********************************************* Warming up GPU!!!"<<endl;
    // Warm up GPU
    warm_up_gpu<<<(10000+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>();

    ////////////////////////////////////

    while (!glfwWindowShouldClose(window)) {
        // Física de partículas
        updateParticlesCUDA(particles, gravity);

        // Detección de colisiones
        collideParticlesCUDA(particles, minDist);

        // Actualizar VBO
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * sizeof(Particle), particles.data());

        // Render
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, particles.size());

        //////////Calculo FPS
        double currentTime = glfwGetTime();
        frameCount++;
        if (currentTime - lastTime >= 0.25) { // Actualizar cada 0.25 segundos para mayor fluidez
            double currentFPS = frameCount / (currentTime - lastTime);
            fps = (fpsAlpha * currentFPS) + ((1.0 - fpsAlpha) * fps); // Suavizado exponencial
            std::string title = "FPS: " + std::to_string((int)fps);
            glfwSetWindowTitle(window, title.c_str());
            frameCount = 0;
            lastTime = currentTime;
        }
        /////////////////////

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
