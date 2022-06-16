#include "PathTracer.h"
#include "massert.h"

#define DEBUG

cudaError_t launchKernel(int* c, const int* a, const int* b, unsigned int size);

void debug_raytracer(dFilm* film, dScene* scene, cudaArray_const_t array, cudaEvent_t event, cudaStream_t stream);

// NOTE: this code is smelly! data clumps/long parameter list -> some kind of struct to contain pathtracing crap?
void wavefront_pathtrace(Paths* paths, Queues* queues, dFilm* film, float4* accumulatebuffer, int* n_samples, int path_count, int max_path_length, dScene* scene, cudaArray_const_t array, cudaEvent_t event, cudaStream_t stream, int tile_size, int tile_x, int tile_y, int max_samples, bool* completed_pixels, uint32_t* nmb_completed_pixels);

void wavefront_init(Paths* paths, int path_count);

PathTracer::PathTracer(int w, int h)
{
    width = w;
    height = h;

    //path_count = w*h*5;
    path_count = tile_size * tile_size * 10;

    cudaMalloc(&accumulatebuffer, width * height * sizeof(float4));
    cudaMalloc(&n_samples, width * height * sizeof(int));
    cudaMalloc(&completed_pixels, width * height * sizeof(bool));

    checkCudaErrors(cudaMallocManaged((void**)&nmb_completed_pixels, sizeof(uint32_t)), "CUDA ERROR: failed to allocate CUDA device memory for queues.");

    *nmb_completed_pixels = 0;

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

    nmb_tile_cols = (width / tile_size);
    nmb_tile_rows = (height / tile_size);

    nmb_tiles = nmb_tile_cols * nmb_tile_rows - 1;
}

void PathTracer::draw_debug(dScene* s)
{
    m_assert(initialized, "draw: pathtracer has not been properly initialized.");

    if (s->update()) {
        clear_buffer();
    }
    cudaArray_t cuda_array;

    interop->get_size(&width, &height);
    checkCudaErrors(interop->map(stream));

    debug_raytracer(d_film, s, interop->array_get(), event, stream);

    checkCudaErrors(interop->unmap(stream));

    interop->blit();
    interop->swap();
}

void PathTracer::render_image(dScene* s)
{
    m_assert(initialized, "draw: pathtracer has not been properly initialized.");

    if (s->update()) {
        reset_image();
    }

    if (*nmb_completed_pixels >= tile_size * tile_size) {

        if (tile_id == nmb_tiles) {
            image_complete = true;
        }

        if (tile_id < nmb_tiles) {
            tile_id++;
        }
        *nmb_completed_pixels = 0;
    }

    tile_x = tile_id / nmb_tile_cols;
    tile_y = tile_id % nmb_tile_rows;

    cudaArray_t cuda_array;

    interop->get_size(&width, &height);
    checkCudaErrors(interop->map(stream));

    if (!image_complete) {
        wavefront_pathtrace(paths, queues, d_film, accumulatebuffer, n_samples, path_count, MAX_PATH_LENGTH, s, interop->array_get(), event, stream, tile_size, tile_x, tile_y, max_samples, completed_pixels, nmb_completed_pixels);
        clear_queues();
    }

    checkCudaErrors(interop->unmap(stream));

    interop->blit();
}

void PathTracer::draw(dScene* s)
{
    m_assert(initialized, "draw: pathtracer has not been properly initialized.");

    if (s->update()) {
        clear_buffer();
    }

    cudaArray_t cuda_array;

    interop->get_size(&width, &height);
    checkCudaErrors(interop->map(stream));

    wavefront_pathtrace(paths, queues, d_film, accumulatebuffer, n_samples, path_count, MAX_PATH_LENGTH, s, interop->array_get(), event, stream, 1024, 0, 0, 16, completed_pixels, nmb_completed_pixels);
    clear_queues();

    checkCudaErrors(interop->unmap(stream));

    interop->blit();
    //interop->swap();
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

    paths_initialized = true;
}

void PathTracer::init_film()
{
    d_film = new dFilm();
    checkCudaErrors(cudaMallocManaged(&d_film, sizeof(dFilm)), "CUDA ERROR: failed to allocate memory " + "(" + (float)sizeof(dFilm)\1000.f + "kB)" + " for film.");
    update_film();
    film_initialized = true;
}

void PathTracer::reset_image()
{
    clear_buffer();
    tile_id = 0;
    image_complete = false;
}

void PathTracer::clear_buffer()
{
    cudaMemset(accumulatebuffer, 0, width * height * sizeof(float4));
    cudaMemset(n_samples, 0, width * height * sizeof(int));
    cudaMemset(completed_pixels, false, width * height * sizeof(bool));

    *nmb_completed_pixels = 0;

    interop->clear();

    wavefront_init(paths, path_count);
}

void PathTracer::set_tile_size(int size)
{
    tile_size = size;

    path_count = tile_size * tile_size * 10;

    init_queues();
    init_paths();

    nmb_tile_cols = (width / tile_size);
    nmb_tile_rows = (height / tile_size);

    nmb_tiles = nmb_tile_cols * nmb_tile_rows - 1;
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
