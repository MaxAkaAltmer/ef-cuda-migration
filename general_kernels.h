#ifndef GENERAL_KERNELS_H
#define GENERAL_KERNELS_H

#define ENABLE_CUDA_CALCULATIONS

void initKernels();
void destroyKernels();

void* allocDevMemory(size_t size);
void freeDevMemory(void *mem);
void copyToDevMemory(void *dst, void *src, size_t size);
void copyToHostMemory(void *dst, void *src, size_t size);

void synchronize();

void run_kernel_field_solver_eval_fields_from_potential(
        int spat_mesh_x_n_nodes,
        int spat_mesh_y_n_nodes,
        int spat_mesh_z_n_nodes,
        double spat_mesh_x_cell_size,
        double spat_mesh_y_cell_size,
        double spat_mesh_z_cell_size,
        const double *spat_mesh_potential,
        double *spat_mesh_electric_field );

void run_kernel_field_solver_compute_phi_next_at_inner_points(
        int spat_mesh_x_n_nodes,
        int spat_mesh_y_n_nodes,
        int spat_mesh_z_n_nodes,
        double dx,
        double dy,
        double dz,
        const double *spat_mesh_charge_density,
        const double *phi_current,
        double *phi_next );

void run_kernel_field_solver_set_phi_next_at_boundaries(
        int nx,
        int ny,
        int nz,
        const double *phi_current,
        double *phi_nextm );

void run_kernel_field_solver_set_phi_next_at_inner_regions(
        const int *nodes,
        double *phi_next,
        int reg_count,
        int nx,
        int ny,
        int nz,
        const double *potential);

bool run_field_solver_iterative_Jacobi_solutions_converged(
        const double *phi_current,
        const double *phi_next,
        double *diff,
        double *rel_diff,
        int nx,
        int ny,
        int nz);

bool run_kernel_particle_to_mesh_map_weight_particles_charge_to_mesh(
        int spat_mesh_x_n_nodes,
        int spat_mesh_y_n_nodes,
        int spat_mesh_z_n_nodes,
        double spat_mesh_x_cell_size,
        double spat_mesh_y_cell_size,
        double spat_mesh_z_cell_size,
        double *spat_mesh_charge_density,
        const double *sources,
        int source_size);

#endif // GENERAL_KERNELS_H
