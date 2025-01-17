#include "field_solver.h"
#define _USE_MATH_DEFINES
#include <math.h>

Field_solver::Field_solver( Spatial_mesh &spat_mesh,
                            Inner_regions_manager &inner_regions )
{
    nx = spat_mesh.x_n_nodes;
    ny = spat_mesh.y_n_nodes;
    nz = spat_mesh.z_n_nodes;
    dx = spat_mesh.x_cell_size;
    dy = spat_mesh.y_cell_size;
    dz = spat_mesh.z_cell_size;

    phi_cuda_buffer_current = allocDevMemory(nx*ny*nz*sizeof(double));
    phi_cuda_buffer_next = allocDevMemory(nx*ny*nz*sizeof(double));
    electric_field_cuda_buffer = allocDevMemory(nx*ny*nz*sizeof(double)*3);
    charge_density_cuda_buffer = allocDevMemory(nx*ny*nz*sizeof(double));
    diff_cuda_buffer = allocDevMemory(nz*sizeof(double));
    diff_cuda_buffer_rel = allocDevMemory(nz*sizeof(double));

    reg_counter_cuda = 0;
    nodes_array_cuda = nullptr;
    potential_array_cuda = nullptr;

    allocate_current_next_phi();
}

void Field_solver::allocate_current_next_phi()
{
    phi_current.resize( boost::extents[nx][ny][nz] );
    phi_next.resize( boost::extents[nx][ny][nz] );
}

void Field_solver::eval_potential( Spatial_mesh &spat_mesh,
                                   Inner_regions_manager &inner_regions )
{
#ifdef ENABLE_CUDA_CALCULATIONS
    solve_poisson_eqn_Jacobi_cuda( spat_mesh, inner_regions );
#else
    solve_poisson_eqn_Jacobi( spat_mesh, inner_regions );
#endif
}

void Field_solver::solve_poisson_eqn_Jacobi_cuda( Spatial_mesh &spat_mesh,
                                             Inner_regions_manager &inner_regions )
{
    max_Jacobi_iterations = 2000;
    int iter;

    size_t mem_size = size_t(nx)*size_t(ny)*size_t(nz)*sizeof(double);
    copyToDevMemory(phi_cuda_buffer_current,spat_mesh.potential.data(), mem_size);
    copyToDevMemory(charge_density_cuda_buffer,spat_mesh.charge_density.data(),mem_size);

    std::vector<int> temp_regions;
    std::vector<double> temp_potential;
    for( auto &reg : inner_regions.regions ){
        for( auto &node : reg.inner_nodes ){
            // todo: mark nodes at edge during construction
            // if (!node.at_domain_edge( nx, ny, nz )) {
            temp_regions.push_back(node.x);
            temp_regions.push_back(node.y);
            temp_regions.push_back(node.z);
            temp_potential.push_back(reg.potential);
            // }
        }
    }

    if(reg_counter_cuda < temp_potential.size())
    {
        if(nodes_array_cuda)
            freeDevMemory(nodes_array_cuda);
        if(potential_array_cuda)
            freeDevMemory(potential_array_cuda);

        reg_counter_cuda = temp_potential.size();
        nodes_array_cuda = allocDevMemory(temp_regions.size()*sizeof(int));
        potential_array_cuda = allocDevMemory(temp_potential.size()*sizeof(double));
    }

    if(temp_potential.size())
    {
        copyToDevMemory(nodes_array_cuda, temp_regions.data(), temp_regions.size()*sizeof(int));
        copyToDevMemory(potential_array_cuda, temp_potential.data(), temp_potential.size()*sizeof(double));
    }

    for( iter = 0; iter < max_Jacobi_iterations; ++iter )
    {
        run_kernel_field_solver_set_phi_next_at_boundaries(
                nx, ny, nz,
                reinterpret_cast<double*>(phi_cuda_buffer_current),
                reinterpret_cast<double*>(phi_cuda_buffer_next) );

        run_kernel_field_solver_compute_phi_next_at_inner_points(
                nx, ny, nz,
                dx, dy, dz,
                reinterpret_cast<double*>(charge_density_cuda_buffer),
                reinterpret_cast<double*>(phi_cuda_buffer_current),
                reinterpret_cast<double*>(phi_cuda_buffer_next) );

        if(temp_potential.size())
        {
            run_kernel_field_solver_set_phi_next_at_inner_regions(
                    reinterpret_cast<int*>(nodes_array_cuda),
                    reinterpret_cast<double*>(phi_cuda_buffer_next),
                    reg_counter_cuda,
                    nx, ny, nz,
                    reinterpret_cast<double*>(potential_array_cuda));
        }

        bool sol = run_field_solver_iterative_Jacobi_solutions_converged(
                reinterpret_cast<double*>(phi_cuda_buffer_current),
                reinterpret_cast<double*>(phi_cuda_buffer_next),
                reinterpret_cast<double*>(diff_cuda_buffer),
                reinterpret_cast<double*>(diff_cuda_buffer_rel),
                nx, ny, nz);

        if ( sol )
        {
            break;
        }

        std::swap( phi_cuda_buffer_current, phi_cuda_buffer_next );
    }

    if ( iter == max_Jacobi_iterations ){
        printf("WARING: potential evaluation did't converge after max iterations!\n");
    }

    //похоже здесь ошибка (я следовал оригиналу) - мы должны брать текущий из-за свапа
    copyToHostMemory(spat_mesh.potential.data(), phi_cuda_buffer_next, mem_size);

    return;
}

void Field_solver::solve_poisson_eqn_Jacobi( Spatial_mesh &spat_mesh,
                                             Inner_regions_manager &inner_regions )
{
    max_Jacobi_iterations = 2000;
    int iter;

    init_current_phi_from_spat_mesh_phi( spat_mesh );
    for( iter = 0; iter < max_Jacobi_iterations; ++iter ){
        single_Jacobi_iteration( spat_mesh, inner_regions );
        if ( iterative_Jacobi_solutions_converged() ) {
            break;
        }
        set_phi_next_as_phi_current();
    }
    if ( iter == max_Jacobi_iterations ){
        printf("WARING: potential evaluation did't converge after max iterations!\n");
    }
    transfer_solution_to_spat_mesh( spat_mesh );

    return;
}


void Field_solver::init_current_phi_from_spat_mesh_phi( Spatial_mesh &spat_mesh )
{
    phi_current.assign( spat_mesh.potential.data(),
                        spat_mesh.potential.data() + spat_mesh.potential.num_elements() );
    return;
}

void Field_solver::single_Jacobi_iteration( Spatial_mesh &spat_mesh,
					    Inner_regions_manager &inner_regions )
{
    set_phi_next_at_boundaries();
    compute_phi_next_at_inner_points( spat_mesh );
    set_phi_next_at_inner_regions( inner_regions );
}

void Field_solver::set_phi_next_at_boundaries()
{
    for ( int j = 0; j < ny; j++ ) {
        for ( int k = 0; k < nz; k++ ) {
            phi_next[0][j][k] = phi_current[0][j][k];
            phi_next[nx-1][j][k] = phi_current[nx-1][j][k];
        }
    }
    //
    for ( int i = 0; i < nx; i++ ) {
        for ( int k = 0; k < nz; k++ ) {
            phi_next[i][0][k] = phi_current[i][0][k];
            phi_next[i][ny-1][k] = phi_current[i][ny-1][k];
        }
    }
    //
    for ( int i = 0; i < nx; i++ ) {
        for ( int j = 0; j < ny; j++ ) {
            phi_next[i][j][0] = phi_current[i][j][0];
            phi_next[i][j][nz-1] = phi_current[i][j][nz-1];
        }
    }
}

void Field_solver::compute_phi_next_at_inner_points( Spatial_mesh &spat_mesh )
{
    double dxdxdydy = dx * dx * dy * dy;
    double dxdxdzdz = dx * dx * dz * dz;
    double dydydzdz = dy * dy * dz * dz;
    double dxdxdydydzdz = dx * dx * dy * dy * dz * dz;
    double denom = 2 * ( dxdxdydy + dxdxdzdz + dydydzdz );
    //
    for ( int i = 1; i < nx - 1; i++ ) {
        for ( int j = 1; j < ny - 1; j++ ) {
            for ( int k = 1; k < nz - 1; k++ ) {
                phi_next[i][j][k] =
                    ( phi_current[i-1][j][k] + phi_current[i+1][j][k] ) * dydydzdz;
                phi_next[i][j][k] = phi_next[i][j][k] +
                    ( phi_current[i][j-1][k] + phi_current[i][j+1][k] ) * dxdxdzdz;
                phi_next[i][j][k] = phi_next[i][j][k] +
                    ( phi_current[i][j][k-1] + phi_current[i][j][k+1] ) * dxdxdydy;
                // Delta phi = - 4 * pi * rho
                phi_next[i][j][k] = phi_next[i][j][k] +
                    4.0 * M_PI * spat_mesh.charge_density[i][j][k] * dxdxdydydzdz;
                phi_next[i][j][k] = phi_next[i][j][k] / denom;
            }
        }
    }
}

void Field_solver::set_phi_next_at_inner_regions( Inner_regions_manager &inner_regions )
{
    for( auto &reg : inner_regions.regions ){
        for( auto &node : reg.inner_nodes ){
            // todo: mark nodes at edge during construction
            // if (!node.at_domain_edge( nx, ny, nz )) {
            phi_next[node.x][node.y][node.z] = reg.potential;
            // }
        }
    }
}


bool Field_solver::iterative_Jacobi_solutions_converged()
{
    // todo: bind tol to config parameters
    //abs_tolerance = std::max( dx * dx, std::max( dy * dy, dz * dz ) ) / 5;
    abs_tolerance = 1.0e-5;
    rel_tolerance = 1.0e-12;
    double diff;
    double rel_diff;
    //double tol;
    //
    for ( int i = 0; i < nx; i++ ) {
        for ( int j = 0; j < ny; j++ ) {
            for ( int k = 0; k < nz; k++ ) {
                diff = fabs( phi_next[i][j][k] - phi_current[i][j][k] );
                rel_diff = diff / fabs( phi_current[i][j][k] );
                if ( diff > abs_tolerance || rel_diff > rel_tolerance ){
                    return false;
                }
            }
        }
    }
    return true;
}


void Field_solver::set_phi_next_as_phi_current()
{
    // Looks like straightforward assignment
    //   phi_next = phi_current
    // would result in copy.
    // Hopefully, it could be avoided with std::swap
    std::swap( phi_current, phi_next );

}

void Field_solver::transfer_solution_to_spat_mesh( Spatial_mesh &spat_mesh )
{
    spat_mesh.potential.assign( phi_next.data(),
                                phi_next.data() + phi_next.num_elements() );
}


void Field_solver::eval_fields_from_potential( Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    boost::multi_array<double, 3> &phi = spat_mesh.potential;
    double ex, ey, ez;
    //

#ifdef ENABLE_CUDA_CALCULATIONS
    copyToDevMemory(phi_cuda_buffer_current,phi.data(),nx*ny*nz*sizeof(double));
    run_kernel_field_solver_eval_fields_from_potential(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            (double*)phi_cuda_buffer_current,
            (double*)electric_field_cuda_buffer);
    copyToHostMemory(spat_mesh.electric_field.data(),electric_field_cuda_buffer,nx*ny*nz*sizeof(double)*3);
    return;
#endif

    for ( int i = 0; i < nx; i++ ) {
        for ( int j = 0; j < ny; j++ ) {
            for ( int k = 0; k < nz; k++ ) {
                if ( i == 0 ) {
                    ex = - boundary_difference( phi[i][j][k], phi[i+1][j][k], dx );
                } else if ( i == nx-1 ) {
                    ex = - boundary_difference( phi[i-1][j][k], phi[i][j][k], dx );
                } else {
                    ex = - central_difference( phi[i-1][j][k], phi[i+1][j][k], dx );
                }

                if ( j == 0 ) {
                    ey = - boundary_difference( phi[i][j][k], phi[i][j+1][k], dy );
                } else if ( j == ny-1 ) {
                    ey = - boundary_difference( phi[i][j-1][k], phi[i][j][k], dy );
                } else {
                    ey = - central_difference( phi[i][j-1][k], phi[i][j+1][k], dy );
                }

                if ( k == 0 ) {
                    ez = - boundary_difference( phi[i][j][k], phi[i][j][k+1], dz );
                } else if ( k == nz-1 ) {
                    ez = - boundary_difference( phi[i][j][k-1], phi[i][j][k], dz );
                } else {
                    ez = - central_difference( phi[i][j][k-1], phi[i][j][k+1], dz );
                }

                spat_mesh.electric_field[i][j][k] = vec3d_init( ex, ey, ez );
            }
        }
    }

    return;
}

double Field_solver::central_difference( double phi1, double phi2, double dx )
{
    return ( (phi2 - phi1) / ( 2.0 * dx ) );
}

double Field_solver::boundary_difference( double phi1, double phi2, double dx )
{
    return ( (phi2 - phi1) / dx );
}


Field_solver::~Field_solver()
{
    freeDevMemory(phi_cuda_buffer_current);
    freeDevMemory(phi_cuda_buffer_next);
    freeDevMemory(electric_field_cuda_buffer);
    freeDevMemory(charge_density_cuda_buffer);
    freeDevMemory(diff_cuda_buffer);
    freeDevMemory(diff_cuda_buffer_rel);

    freeDevMemory(nodes_array_cuda);
    freeDevMemory(potential_array_cuda);

    // delete phi arrays?
}
