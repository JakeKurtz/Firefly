#include "PathTracer.h"
#include "massert.h"

#define DEBUG

cudaError_t launchKernel(int* c, const int* a, const int* b, unsigned int size);

void debug_raytracer(dFilm* film, dScene* scene, cudaArray_const_t array, cudaEvent_t event, cudaStream_t stream);

PathTracer::PathTracer(int w, int h)
{
    width = w;
    height = h;

    cudaMalloc(&accumulatebuffer, width * height * sizeof(float4));
    cudaMalloc(&n_samples, width * height * sizeof(int));

    init_interop();
    init_queues();
    init_paths();
    init_film();

    initialized = (
        interop_initialized &&
        queues_initialized &&
        paths_initialized &&
        film_initialized
    );
}

void PathTracer::draw(dScene* s)
{
    m_assert(initialized, "draw: pathtracer has not been properly initialized.");

    s->update();

    cudaArray_t cuda_array;

    interop->get_size(&width, &height);
    checkCudaErrors(interop->map(stream));

    debug_raytracer(d_film, s, interop->array_get(), event, stream);

    checkCudaErrors(interop->unmap(stream));

    interop->blit();
    interop->swap();

    //frame++;
}

void PathTracer::init_interop()
{
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));   // optionally ignore default stream behavior
    checkCudaErrors(cudaEventCreateWithFlags(&event, cudaEventBlockingSync)); // | cudaEventDisableTiming);

    // Testing: DO NOT SET TO FALSE, ONLY TRUE IS RELIABLE
    try {
        interop = new Interop(true, 2);
        checkCudaErrors(interop->set_size(width, height));

        interop_initialized = true;
    }
    catch(const std::exception& e){
       std::cerr << e.what() << std::endl;
    }
}

void PathTracer::init_queues()
{
    queues_initialized = true;
}

void PathTracer::init_paths()
{
    paths_initialized = true;
}

void PathTracer::init_film()
{
    d_film = new dFilm();
    checkCudaErrors(cudaMallocManaged(&d_film, sizeof(dFilm)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(dFilm)\1000.f + "kB)" + " for film.");
    update_film();
    film_initialized = true;
}

void PathTracer::clear_buffer()
{
}

void PathTracer::clear_queues()
{
}

void PathTracer::clear_paths()
{
}

void PathTracer::update_film()
{
    d_film->gamma = 1.f;
    d_film->hres = height;
    d_film->vres = width;
    //checkCudaErrors(cudaMemcpyToSymbol(, &h_film, sizeof(Film*)));
}
