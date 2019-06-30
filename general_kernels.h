#ifndef GENERAL_KERNELS_H
#define GENERAL_KERNELS_H

void initKernels();
void destroyKernels();

void* allocDevMemory(size_t size);
void freeDevMemory(void *mem);
void copyToDevMemory(void *dst, void *src, size_t size);
void copyToHostMemory(void *dst, void *src, size_t size);

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

#endif // GENERAL_KERNELS_H
