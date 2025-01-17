#include "particle_to_mesh_map.h"

// Eval charge density on grid
void Particle_to_mesh_map::weight_particles_charge_to_mesh( 
    Spatial_mesh &spat_mesh, Particle_sources_manager &particle_sources  )
{
#ifdef ENABLE_CUDA_CALCULATIONS
    size_t source_size;
    void *sources = particle_sources.getCudaSourcesChargeNPos(source_size);

    size_t charge_size = spat_mesh.x_n_nodes * spat_mesh.y_n_nodes * spat_mesh.z_n_nodes *sizeof(double);
    if(charge_size > cuda_spat_mesh_charge_density_size)
    {
        freeDevMemory(cuda_spat_mesh_charge_density);
        cuda_spat_mesh_charge_density = allocDevMemory(charge_size);
        cuda_spat_mesh_charge_density_size = charge_size;
    }

    run_kernel_particle_to_mesh_map_weight_particles_charge_to_mesh(
            spat_mesh.x_n_nodes,
            spat_mesh.y_n_nodes,
            spat_mesh.z_n_nodes,
            spat_mesh.x_cell_size,
            spat_mesh.y_cell_size,
            spat_mesh.z_cell_size,
            (double*)cuda_spat_mesh_charge_density,
            (double*)sources,
            source_size);

    copyToHostMemory(spat_mesh.charge_density.data(),cuda_spat_mesh_charge_density,charge_size);

    return;
#endif
    // Rewrite:
    // forall particles {
    //   find nonzero weights and corresponding nodes
    //   charge[node] = weight(particle, node) * particle.charge
    // }
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    double cell_volume = dx * dy * dz;
    double volume_around_node = cell_volume;
    int tlf_i, tlf_j, tlf_k; // 'tlf' = 'top_left_far'
    double tlf_x_weight, tlf_y_weight, tlf_z_weight;

    for( auto& part_src: particle_sources.sources ) {
        for( auto& p : part_src.particles ) {
            next_node_num_and_weight( vec3d_x( p.position ), dx, &tlf_i, &tlf_x_weight );
            next_node_num_and_weight( vec3d_y( p.position ), dy, &tlf_j, &tlf_y_weight );
            next_node_num_and_weight( vec3d_z( p.position ), dz, &tlf_k, &tlf_z_weight );
            spat_mesh.charge_density[tlf_i][tlf_j][tlf_k] +=
            tlf_x_weight * tlf_y_weight * tlf_z_weight
            * p.charge / volume_around_node;
            spat_mesh.charge_density[tlf_i-1][tlf_j][tlf_k] +=
            ( 1.0 - tlf_x_weight ) * tlf_y_weight * tlf_z_weight
            * p.charge / volume_around_node;
            spat_mesh.charge_density[tlf_i][tlf_j-1][tlf_k] +=
            tlf_x_weight * ( 1.0 - tlf_y_weight ) * tlf_z_weight
            * p.charge / volume_around_node;
            spat_mesh.charge_density[tlf_i-1][tlf_j-1][tlf_k] +=
            ( 1.0 - tlf_x_weight ) * ( 1.0 - tlf_y_weight ) * tlf_z_weight
            * p.charge / volume_around_node;
            spat_mesh.charge_density[tlf_i][tlf_j][tlf_k - 1] +=
            tlf_x_weight * tlf_y_weight * ( 1.0 - tlf_z_weight )
            * p.charge / volume_around_node;
            spat_mesh.charge_density[tlf_i-1][tlf_j][tlf_k - 1] +=
            ( 1.0 - tlf_x_weight ) * tlf_y_weight * ( 1.0 - tlf_z_weight )
            * p.charge / volume_around_node;
            spat_mesh.charge_density[tlf_i][tlf_j-1][tlf_k - 1] +=
            tlf_x_weight * ( 1.0 - tlf_y_weight ) * ( 1.0 - tlf_z_weight )
            * p.charge / volume_around_node;
            spat_mesh.charge_density[tlf_i-1][tlf_j-1][tlf_k - 1] +=
            ( 1.0 - tlf_x_weight ) * ( 1.0 - tlf_y_weight ) * ( 1.0 - tlf_z_weight )
            * p.charge / volume_around_node;
        }
    }
    return;
}

Vec3d Particle_to_mesh_map::field_at_particle_position( 
    Spatial_mesh &spat_mesh, Particle &p )
{
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    int tlf_i, tlf_j, tlf_k; // 'tlf' = 'top_left_far'
    double tlf_x_weight, tlf_y_weight, tlf_z_weight;  
    Vec3d field_from_node, total_field;
    //
    next_node_num_and_weight( vec3d_x( p.position ), dx, &tlf_i, &tlf_x_weight );
    next_node_num_and_weight( vec3d_y( p.position ), dy, &tlf_j, &tlf_y_weight );
    next_node_num_and_weight( vec3d_z( p.position ), dz, &tlf_k, &tlf_z_weight );
    // tlf
    total_field = vec3d_zero();
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[tlf_i][tlf_j][tlf_k],
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // trf
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[tlf_i-1][tlf_j][tlf_k],
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // blf
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[tlf_i][tlf_j - 1][tlf_k],	
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // brf
    field_from_node = vec3d_times_scalar(			
	spat_mesh.electric_field[tlf_i-1][tlf_j-1][tlf_k],	
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // tln
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[tlf_i][tlf_j][tlf_k-1],
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // trn
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[tlf_i-1][tlf_j][tlf_k-1],
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // bln
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[tlf_i][tlf_j - 1][tlf_k-1],	
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // brn
    field_from_node = vec3d_times_scalar(			
	spat_mesh.electric_field[tlf_i-1][tlf_j-1][tlf_k-1],	
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );    
    //
    return total_field;
}


Vec3d Particle_to_mesh_map::force_on_particle( 
    Spatial_mesh &spat_mesh, Particle &p )
{
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    int tlf_i, tlf_j, tlf_k; // 'tlf' = 'top_left_far'
    double tlf_x_weight, tlf_y_weight, tlf_z_weight;  
    Vec3d field_from_node, total_field, force;
    //
    next_node_num_and_weight( vec3d_x( p.position ), dx, &tlf_i, &tlf_x_weight );
    next_node_num_and_weight( vec3d_y( p.position ), dy, &tlf_j, &tlf_y_weight );
    next_node_num_and_weight( vec3d_z( p.position ), dz, &tlf_k, &tlf_z_weight );
    // tlf
    total_field = vec3d_zero();
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[tlf_i][tlf_j][tlf_k],
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // trf
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[tlf_i-1][tlf_j][tlf_k],
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // blf
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[tlf_i][tlf_j - 1][tlf_k],	
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // brf
    field_from_node = vec3d_times_scalar(			
	spat_mesh.electric_field[tlf_i-1][tlf_j-1][tlf_k],	
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // tln
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[tlf_i][tlf_j][tlf_k-1],
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // trn
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[tlf_i-1][tlf_j][tlf_k-1],
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // bln
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[tlf_i][tlf_j - 1][tlf_k-1],	
	tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // brn
    field_from_node = vec3d_times_scalar(			
	spat_mesh.electric_field[tlf_i-1][tlf_j-1][tlf_k-1],	
	1.0 - tlf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - tlf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );    
    //
    force = vec3d_times_scalar( total_field, p.charge );
    return force;
}

void Particle_to_mesh_map::next_node_num_and_weight( 
    const double x, const double grid_step, 
    int *next_node, double *weight )
{
    double x_in_grid_units = x / grid_step;
    *next_node = ceil( x_in_grid_units );
    *weight = 1.0 - ( *next_node - x_in_grid_units );
    return;
}
