#include "PathTracer.h"
#include "massert.h"

#define DEBUG

cudaError_t launchKernel(int* c, const int* a, const int* b, unsigned int size);

void debug_raytracer(dFilm* film, dScene* scene, cudaArray_const_t array, cudaEvent_t event, cudaStream_t stream);

// NOTE: this code is smelly! data clumps/long parameter list -> some kind of struct to contain pathtracing crap?
void wavefront_pathtrace(Paths* paths, Queues* queues, dFilm* film, float4* accumulatebuffer, int* n_samples, int path_count, int max_path_length, dScene* scene, cudaArray_const_t array, cudaEvent_t event, cudaStream_t stream);

void wavefront_init(Paths* paths, int path_count);

PathTracer::PathTracer(int w, int h)
{
    width = w;
    height = h;

    path_count = width * height * 1;

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

void PathTracer::draw_debug(dScene* s)
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

void PathTracer::draw(dScene* s)
{
    m_assert(initialized, "draw: pathtracer has not been properly initialized.");

    s->update();

    cudaArray_t cuda_array;

    interop->get_size(&width, &height);
    checkCudaErrors(interop->map(stream));

    wavefront_pathtrace(paths, queues, d_film, accumulatebuffer, n_samples, path_count, MAX_PATH_LENGTH, s, interop->array_get(), event, stream);
    clear_queues();

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
    checkCudaErrors(cudaMallocManaged(&queues, sizeof(Queues)), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&queues->queue_newPath, sizeof(uint32_t) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&queues->queue_mat_diffuse, sizeof(uint32_t) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&queues->queue_mat_cook, sizeof(uint32_t) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&queues->queue_mat_mix, sizeof(uint32_t) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&queues->queue_extension, sizeof(uint32_t) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&queues->queue_shadow, sizeof(uint32_t) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");

    queues_initialized = true;
}

void PathTracer::init_paths()
{
    checkCudaErrors(cudaMallocManaged(&paths, sizeof(Paths)), "CUDA ERROR: failed to allocate CUDA device memory for paths.");
    checkCudaErrors(cudaMallocManaged(&paths->film_pos, sizeof(float2) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->throughput, sizeof(float3) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->length, sizeof(uint32_t) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->ext_ray, sizeof(dRay) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->ext_isect, sizeof(Isect) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->ext_brdf, sizeof(float3) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->ext_brdf_type, sizeof(BSDF) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->ext_pdf, sizeof(float) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->ext_cosine, sizeof(float) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->ext_specular, sizeof(bool) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->light_ray, sizeof(dRay) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->light_id, sizeof(int) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->light_emittance, sizeof(float3) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->light_brdf, sizeof(float3) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->light_samplePoint, sizeof(float3) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->light_pdf, sizeof(float) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->light_cosine, sizeof(float) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");
    checkCudaErrors(cudaMallocManaged(&paths->light_visible, sizeof(bool) * path_count), "CUDA ERROR: failed to allocate CUDA device memory for queues.");

    wavefront_init(paths, path_count);

    //wf_init <<< GRIDSIZE, BLOCKSIZE >>> (paths, path_count);
    //checkCudaErrors(cudaGetLastError(), "CUDA ERROR: failed to initialize paths.");
    //checkCudaErrors(cudaDeviceSynchronize());

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
    //if (buffer_reset) {
    //    cameraWasMoving = true;
    //}

    //if (cameraWasMoving) {
        cudaMemset(accumulatebuffer, 0, width * height * sizeof(float4));
        cudaMemset(n_samples, 0, width * height * sizeof(int));
        //frame = 0;
        interop->clear();
    //    clear_counter--;
    //}

    //if (clear_counter == 0) {
    //    cameraWasMoving = false;
    //    clear_counter = 5;
    //}

    //buffer_reset = false;
}

void PathTracer::clear_queues()
{
    queues->queue_newPath_length = 0;
    queues->queue_mat_diffuse_length = 0;
    queues->queue_mat_cook_length = 0;
    queues->queue_mat_mix_length = 0;
    queues->queue_extension_length = 0;
    queues->queue_shadow_length = 0;
}

void PathTracer::clear_paths()
{
}

void PathTracer::update_film()
{
    d_film->gamma = 1.f;
    d_film->hres = width;
    d_film->vres = height;
    //checkCudaErrors(cudaMemcpyToSymbol(, &h_film, sizeof(Film*)));
}
