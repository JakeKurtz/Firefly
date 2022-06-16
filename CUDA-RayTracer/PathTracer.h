#pragma once

#include "CudaHelpers.h"

#include "dScene.h"
#include "Interop.h"

#include <iostream>
#include "Wavefront.cuh"

class PathTracer
{
public:
    PathTracer(int w, int h);
    void draw(dScene* s);
    void render_image(dScene* s);
    void draw_debug(dScene* s);

    void reset_image();
    void clear_buffer();

    void set_tile_size(int size);

    Interop* interop;

    uint32_t* nmb_completed_pixels;

    int max_samples = 1;
    int tile_size = 256;
    int tile_id = 0;

private:
    const int BLOCK_SIZE = 64;
    const uint32_t  MAX_PATH_LENGTH = 5;

    int width;
    int height;

    int path_count = 0;
    int grid_size = 0;

    __constant__ Paths* paths;
    __constant__ Queues* queues;

    // image buffer storing accumulated pixel samples
    float4* accumulatebuffer;
    int* n_samples;

    bool* completed_pixels = 0;

    dScene* scene;
    dFilm* d_film = nullptr;
    dCamera* d_camera = nullptr;

    cudaStream_t stream;
    cudaEvent_t  event;

    bool interop_initialized = false;
    bool queues_initialized = false;
    bool paths_initialized = false;
    bool film_initialized = false;
    bool initialized = false;

    bool image_complete = false;

    int nmb_tiles = 0;

    int nmb_tile_cols = 0;
    int nmb_tile_rows = 0;

    int tile_x = 0;
    int tile_y = 0;

    int samples = 0;

    void init_interop();
    void init_queues();
    void init_paths();
    void init_film();

    void update_film();

    void clear_queues();
    void clear_paths();

    int get_grid_size() { return grid_size; };
    int get_block_size() { return (path_count + BLOCK_SIZE - 1) / BLOCK_SIZE; };
};

