#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <math.h>

void initKernels()
{
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    assert(gpu_count>0);
    assert(cudaSetDevice(0) == cudaSuccess);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << ">>> Cuda limits: threads(" << prop.maxThreadsPerBlock << ") "
              << "threads dim(" << prop.maxThreadsDim[0] << "," << prop.maxThreadsDim[1] << "," << prop.maxThreadsDim[2] << ") "
              << std::endl;
}

void destroyKernels()
{

}

void* allocDevMemory(size_t size)
{
    void *mem;
    if(!size) return nullptr;
    assert(cudaMalloc(&mem,size) == cudaSuccess);
    return mem;
}

void freeDevMemory(void *mem)
{
    if(!mem) return;
    assert(cudaStreamSynchronize(0) == cudaSuccess);
    assert(cudaFree(mem) == cudaSuccess);
}

void copyToDevMemory(void *dst, void *src, size_t size)
{
    assert(cudaMemcpy(dst,src,size,cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaStreamSynchronize(0) == cudaSuccess);
}

void synchronize()
{
    assert(cudaStreamSynchronize(0) == cudaSuccess);
}

void copyToHostMemory(void *dst, void *src, size_t size)
{
    assert(cudaStreamSynchronize(0) == cudaSuccess);
    assert(cudaMemcpy(dst,src,size,cudaMemcpyDeviceToHost) == cudaSuccess);
}

static __inline__ __device__ double central_difference( double phi1, double phi2, double dx )
{
    return ( (phi2 - phi1) / ( 2.0 * dx ) );
}

static __inline__ __device__ double boundary_difference( double phi1, double phi2, double dx )
{
    return ( (phi2 - phi1) / dx );
}

static __inline__ __device__ int ceil_index( int ny, int nz, int x, int y, int z )
{
    return x*ny*nz+y*nz+z;
}

static __global__ void kernel_field_solver_eval_fields_from_potential(
        int spat_mesh_x_n_nodes,
        int spat_mesh_y_n_nodes,
        int spat_mesh_z_n_nodes,
        double spat_mesh_x_cell_size,
        double spat_mesh_y_cell_size,
        double spat_mesh_z_cell_size,
        const double *spat_mesh_potential,
        double *spat_mesh_electric_field
        )
{
    int nx = spat_mesh_x_n_nodes;
    int ny = spat_mesh_y_n_nodes;
    int nz = spat_mesh_z_n_nodes;
    double dx = spat_mesh_x_cell_size;
    double dy = spat_mesh_y_cell_size;
    double dz = spat_mesh_z_cell_size;
    const double *phi = spat_mesh_potential;
    double ex, ey, ez;

    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;

    if(i>=nx || j>=ny) return;

    for ( int k = 0; k < nz; k++ )
    {
        if ( i == 0 ) {
            ex = - boundary_difference( phi[ceil_index(ny,nz,i,j,k)], phi[ceil_index(ny,nz,i+1,j,k)], dx );
        } else if ( i == nx-1 ) {
            ex = - boundary_difference( phi[ceil_index(ny,nz,i-1,j,k)], phi[ceil_index(ny,nz,i,j,k)], dx );
        } else {
            ex = - central_difference( phi[ceil_index(ny,nz,i-1,j,k)], phi[ceil_index(ny,nz,i+1,j,k)], dx );
        }

        if ( j == 0 ) {
            ey = - boundary_difference( phi[ceil_index(ny,nz,i,j,k)], phi[ceil_index(ny,nz,i,j+1,k)], dy );
        } else if ( j == ny-1 ) {
            ey = - boundary_difference( phi[ceil_index(ny,nz,i,j-1,k)], phi[ceil_index(ny,nz,i,j,k)], dy );
        } else {
            ey = - central_difference( phi[ceil_index(ny,nz,i,j-1,k)], phi[ceil_index(ny,nz,i,j+1,k)], dy );
        }

        if ( k == 0 ) {
            ez = - boundary_difference( phi[ceil_index(ny,nz,i,j,k)], phi[ceil_index(ny,nz,i,j,k+1)], dz );
        } else if ( k == nz-1 ) {
            ez = - boundary_difference( phi[ceil_index(ny,nz,i,j,k-1)], phi[ceil_index(ny,nz,i,j,k)], dz );
        } else {
            ez = - central_difference( phi[ceil_index(ny,nz,i,j,k-1)], phi[ceil_index(ny,nz,i,j,k+1)], dz );
        }

        spat_mesh_electric_field[ceil_index(ny,nz,i,j,k)*3+0] = ex;
        spat_mesh_electric_field[ceil_index(ny,nz,i,j,k)*3+1] = ey;
        spat_mesh_electric_field[ceil_index(ny,nz,i,j,k)*3+2] = ez;
    }

}

void run_kernel_field_solver_eval_fields_from_potential(
        int spat_mesh_x_n_nodes,
        int spat_mesh_y_n_nodes,
        int spat_mesh_z_n_nodes,
        double spat_mesh_x_cell_size,
        double spat_mesh_y_cell_size,
        double spat_mesh_z_cell_size,
        const double *spat_mesh_potential,
        double *spat_mesh_electric_field
        )
{
    dim3 block(16,16);
    dim3 grid((spat_mesh_x_n_nodes+15)/16,(spat_mesh_y_n_nodes+15)/16);

    kernel_field_solver_eval_fields_from_potential <<<grid,block>>>(spat_mesh_x_n_nodes,
                                                                    spat_mesh_y_n_nodes,
                                                                    spat_mesh_z_n_nodes,
                                                                    spat_mesh_x_cell_size,
                                                                    spat_mesh_y_cell_size,
                                                                    spat_mesh_z_cell_size,
                                                                    spat_mesh_potential,
                                                                    spat_mesh_electric_field);
    //std::cout << "run_kernel_field_solver_eval_fields_from_potential: " << cudaGetLastError() << " " << grid.x << " " << grid.y << " " << grid.z << " " << block.x << " " << block.y << " " << block.z << std::endl;
    assert(cudaGetLastError() == cudaSuccess);
}

static __global__ void kernel_field_solver_compute_phi_next_at_inner_points(
        int nx,
        int ny,
        int nz,
        double dxdxdydy,
        double dxdxdzdz,
        double dydydzdz,
        double dxdxdydydzdz,
        double denom,
        double m_pi,
        const double *spat_mesh_charge_density,
        const double *phi_current,
        double *phi_next
        )
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;

    if(i>=nx || j>=ny) return;

    for ( int k = 1; k < nz - 1; k++ ) {
        phi_next[ceil_index(ny,nz,i,j,k)] =
            ( phi_current[ceil_index(ny,nz,i-1,j,k)] + phi_current[ceil_index(ny,nz,i+1,j,k)] ) * dydydzdz;
        phi_next[ceil_index(ny,nz,i,j,k)] = phi_next[ceil_index(ny,nz,i,j,k)] +
            ( phi_current[ceil_index(ny,nz,i,j-1,k)] + phi_current[ceil_index(ny,nz,i,j+1,k)] ) * dxdxdzdz;
        phi_next[ceil_index(ny,nz,i,j,k)] = phi_next[ceil_index(ny,nz,i,j,k)] +
            ( phi_current[ceil_index(ny,nz,i,j,k-1)] + phi_current[ceil_index(ny,nz,i,j,k+1)] ) * dxdxdydy;
        phi_next[ceil_index(ny,nz,i,j,k)] = phi_next[ceil_index(ny,nz,i,j,k)] +
            4.0 * m_pi * spat_mesh_charge_density[ceil_index(ny,nz,i,j,k)] * dxdxdydydzdz;
        phi_next[ceil_index(ny,nz,i,j,k)] = phi_next[ceil_index(ny,nz,i,j,k)] / denom;
    }
}

void run_kernel_field_solver_compute_phi_next_at_inner_points(
        int spat_mesh_x_n_nodes,
        int spat_mesh_y_n_nodes,
        int spat_mesh_z_n_nodes,
        double dx,
        double dy,
        double dz,
        const double *spat_mesh_charge_density,
        const double *phi_current,
        double *phi_next
        )
{
    double dxdxdydy = dx * dx * dy * dy;
    double dxdxdzdz = dx * dx * dz * dz;
    double dydydzdz = dy * dy * dz * dz;
    double dxdxdydydzdz = dx * dx * dy * dy * dz * dz;
    double denom = 2 * ( dxdxdydy + dxdxdzdz + dydydzdz );

    dim3 block(16,16);
    dim3 grid((spat_mesh_x_n_nodes+15)/16,(spat_mesh_y_n_nodes+15)/16);

    kernel_field_solver_compute_phi_next_at_inner_points <<<grid,block>>>(spat_mesh_x_n_nodes,
                                                                          spat_mesh_y_n_nodes,
                                                                          spat_mesh_z_n_nodes,
                                                                          dxdxdydy,
                                                                          dxdxdzdz,
                                                                          dydydzdz,
                                                                          dxdxdydydzdz,
                                                                          denom,
                                                                          3.14159265358979323846,
                                                                          spat_mesh_charge_density,
                                                                          phi_current,
                                                                          phi_next);

    //std::cout << "run_kernel_field_solver_compute_phi_next_at_inner_points: " << cudaGetLastError() << " " << grid.x << " " << grid.y << " " << grid.z << " " << block.x << " " << block.y << " " << block.z << std::endl;
    assert(cudaGetLastError() == cudaSuccess);
}

static __global__ void kernel_field_solver_set_phi_next_at_boundaries_ny_nz(
        int nx,
        int ny,
        int nz,
        const double *phi_current,
        double *phi_next)
{
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    int k = blockIdx.y*blockDim.y+threadIdx.y;

    if(j>=ny || k>nz) return;

    phi_next[ceil_index(ny,nz,0,j,k)] = phi_current[ceil_index(ny,nz,0,j,k)];
    phi_next[ceil_index(ny,nz,nx-1,j,k)] = phi_current[ceil_index(ny,nz,nx-1,j,k)];
}

static __global__ void kernel_field_solver_set_phi_next_at_boundaries_nx_nz(
        int nx,
        int ny,
        int nz,
        const double *phi_current,
        double *phi_next)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int k = blockIdx.y*blockDim.y+threadIdx.y;

    if(i>=nx || k>nz) return;

    phi_next[ceil_index(ny,nz,i,0,k)] = phi_current[ceil_index(ny,nz,i,0,k)];
    phi_next[ceil_index(ny,nz,i,ny-1,k)] = phi_current[ceil_index(ny,nz,i,ny-1,k)];
}

static __global__ void kernel_field_solver_set_phi_next_at_boundaries_nx_ny(
        int nx,
        int ny,
        int nz,
        const double *phi_current,
        double *phi_next)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;

    if(i>=nx || j>=ny) return;

    phi_next[ceil_index(ny,nz,i,j,0)] = phi_current[ceil_index(ny,nz,i,j,0)];
    phi_next[ceil_index(ny,nz,i,j,nz-1)] = phi_current[ceil_index(ny,nz,i,j,nz-1)];
}

void run_kernel_field_solver_set_phi_next_at_boundaries(int nx,
                                                        int ny,
                                                        int nz,
                                                        const double *phi_current,
                                                        double *phi_next)
{
    dim3 block(16,16);
    dim3 grid((ny+15)/16,(nz+15)/16);
    kernel_field_solver_set_phi_next_at_boundaries_ny_nz<<<grid,block>>>(nx,ny,nz,phi_current,phi_next);
    assert(cudaGetLastError() == cudaSuccess);

    grid = dim3((nx+15)/16,(nz+15)/16);
    kernel_field_solver_set_phi_next_at_boundaries_nx_nz<<<grid,block>>>(nx,ny,nz,phi_current,phi_next);
    assert(cudaGetLastError() == cudaSuccess);

    grid = dim3((nx+15)/16,(ny+15)/16);
    kernel_field_solver_set_phi_next_at_boundaries_nx_ny<<<grid,block>>>(nx,ny,nz,phi_current,phi_next);
    assert(cudaGetLastError() == cudaSuccess);
}

static __global__ void kernel_field_solver_set_phi_next_at_inner_regions(
        const int *nodes,
        double *phi_next,
        int reg_count,
        int nx,
        int ny,
        int nz,
        const double *potential)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(reg_count <= i) return;

    // todo: mark nodes at edge during construction
    // if (!node.at_domain_edge( nx, ny, nz )) {
    phi_next[ceil_index(ny,nz,nodes[i*3],nodes[i*3+1],nodes[i*3+2])] = potential[i];
    // }
}

void run_kernel_field_solver_set_phi_next_at_inner_regions(
        const int *nodes,
        double *phi_next,
        int reg_count,
        int nx,
        int ny,
        int nz,
        const double *potential)
{
    dim3 block(256);
    dim3 grid((reg_count+255)/256);

    kernel_field_solver_set_phi_next_at_inner_regions<<<grid,block>>>(nodes,phi_next,reg_count,nx,ny,nz,potential);
    assert(cudaGetLastError() == cudaSuccess);
}

static __global__ void kernel_feld_solver_iterative_Jacobi_solutions_converged(
        const double *phi_current,
        const double *phi_next,
        double *diff,
        double *rel_diff,
        int nx,
        int ny,
        int nz)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;

    if(i>=nx || j>=ny) return;

    for ( int k = 0; k < nz; k++ )
    {
        double d = abs( phi_next[ceil_index(ny,nz,i,j,k)] - phi_current[ceil_index(ny,nz,i,j,k)] );
        double rd = d / abs( phi_current[ceil_index(ny,nz,i,j,k)] );
        if(!k || d>diff[k]) diff[k] = d;
        if(!k || rd>rel_diff[k]) rel_diff[k] = rd;
    }
}

bool run_field_solver_iterative_Jacobi_solutions_converged(
        const double *phi_current,
        const double *phi_next,
        double *diff,
        double *rel_diff,
        int nx,
        int ny,
        int nz)
{
    // todo: bind tol to config parameters
    //abs_tolerance = std::max( dx * dx, std::max( dy * dy, dz * dz ) ) / 5;
    double abs_tolerance = 1.0e-5;
    double rel_tolerance = 1.0e-12;
    //double tol;
    //

    return false;

    dim3 block(16,16);
    dim3 grid((nx+15)/16,(ny+15)/16);

    kernel_feld_solver_iterative_Jacobi_solutions_converged <<<grid,block>>> (phi_current, phi_next, diff, rel_diff, nx, ny, nz);
    assert(cudaGetLastError() == cudaSuccess);

    std::vector<double> maximum_diff, maximum_rel_diff;
    maximum_diff.resize(nz);
    maximum_rel_diff.resize(nz);

    copyToHostMemory(maximum_diff.data(), diff, sizeof(double)*nz);
    copyToHostMemory(maximum_rel_diff.data(), rel_diff, sizeof(double)*nz);

    for(int i=0;i<nz;i++)
    {
        if ( maximum_diff[i] > abs_tolerance || maximum_rel_diff[i] > rel_tolerance ){
            return false;
        }
    }

    return true;
}
