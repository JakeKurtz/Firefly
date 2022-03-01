#pragma once

#include "CudaHelpers.h"

#include "dScene.h"
#include "Interop.h"

#include <iostream>

class PathTracer
{
public:
    PathTracer(int w, int h);
    void draw(dScene* s);
private:
    int             width;
    int             height;

    uint32_t        PATHCOUNT = width * height;
    const uint32_t  MAXPATHLENGTH = 2;

    const int       BLOCKSIZE = 64;
    const int       GRIDSIZE = (PATHCOUNT + BLOCKSIZE - 1) / BLOCKSIZE;

    // image buffer storing accumulated pixel samples
    float4* accumulatebuffer;
    int* n_samples;

    dScene* scene;
    dFilm* d_film = nullptr;
    dCamera* d_camera = nullptr;

    Interop* interop;
    cudaStream_t stream;
    cudaEvent_t  event;

    bool interop_initialized = false;
    bool queues_initialized = false;
    bool paths_initialized = false;
    bool film_initialized = false;
    bool initialized = false;

    void init_interop();
    void init_queues();
    void init_paths();
    void init_film();

    void update_film();

    void clear_buffer();
    void clear_queues();
    void clear_paths();
};

