import os, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

def get_electrons_current( h5file ):
    time_step = h5file["/Time_grid"].attrs["time_step_size"][0]
    charge = h5file["/Particle_sources/electrons"].attrs["charge"][0]
    particles_per_step = h5file[
        "/Particle_sources/electrons"].attrs["particles_to_generate_each_step"][0]
    current = particles_per_step * charge / time_step
    return current

def get_electrons_current_density( h5file ):
    I = get_electrons_current( h5file )
    radius = h5file["/Particle_sources/electrons"].attrs["cylinder_radius"][0]
    current_density = I / ( np.pi * radius * radius )
    return current_density

def get_electrons_energy( h5file ):    
    pz = h5file["/Particle_sources/electrons"].attrs["mean_momentum_z"][0]
    m = h5file["/Particle_sources/electrons"].attrs["mass"][0]
    E = pz * pz / ( 2.0 * m )
    return E

def get_ions_parameters( h5file ):
    mass = h5file["/Particle_sources/ions"].attrs["mass"][0]
    charge = h5file["/Particle_sources/ions"].attrs["charge"][0]
    return ( mass, charge )

def eq10_test( j, xi, A, E_e ):
    # j - beam current density [ Ampere / cm^2 ]
    # xi - charge state of the ion
    # A - atomic weight of the ion
    # E_e - electron energy [keV] !!!
    return 5.8e6 * j * xi * xi / A / np.sqrt( E_e * E_e * E_e )

def eq9_test( m_electron, e, v_e, n_e, M_ion, xi ):
    kT = 1.380e-16 * 3000
    p_0 = xi * e * e / m_electron / v_e ** 2
    debaye_len = np.sqrt( kT / ( 4 * np.pi * n_e * e * e ) )
    Lambda = debaye_len / p_0
    tmp1 = 2.0 * np.pi * m_electron / M_ion
    tmp2 = n_e * v_e ** 2 * p_0 ** 2
    return tmp1 * tmp2 * np.log( 1 + Lambda ** 2 )


def find_necessary_out_files():
    os.chdir("./")
    h5files = []
    for file in glob.glob("thermolization_[0-9]*.h5"):
        h5files.append( file )
    return h5files


### Plot as points 

out_files = find_necessary_out_files()
out_files.sort()
print( out_files )

h5file = h5py.File( out_files[0], mode = "r" )
I = get_electrons_current( h5file )
j = get_electrons_current_density( h5file )
E_e = get_electrons_energy( h5file )
m, q = get_ions_parameters( h5file )
z_volume_size = h5file["/Spatial_mesh"].attrs["z_volume_size"][0]
x_volume_size = h5file["/Spatial_mesh"].attrs["x_volume_size"][0]

### V_prll
xi = 1
A = 1
ampere_to_cgs = 2997924536.8431
j_ampere = j / ampere_to_cgs
ev_to_cgs = 1.60218e-12
E_e_keV = E_e / ev_to_cgs / 1000
a_prll = eq10_test( np.abs(j_ampere), xi, A, E_e_keV )
v_prll_initial = 0 # todo: read from config

plt.figure()
plt.xlabel( "V_prll [cm/s]" )
plt.ylabel( "Time" )
yticks = range( len(out_files) )
yticks_label = []

# todo: use time instead of time_step_num
for time_step_num, filename in enumerate(out_files):
    h5file = h5py.File( filename, mode = "r" )
    p_z_num = h5file["/Particle_sources/ions/momentum_z"]
    v_prll = p_z_num / m

    time_step = h5file["/Time_grid"].attrs["current_time"][0]
    yticks_label.append( time_step )
    
    y = np.full_like( v_prll, time_step_num )
    plt.plot( v_prll, y, marker = '.', markersize = 10, color = "g", alpha = 0.005 )

    v_prll_mean = np.mean( v_prll )
    v_prll_stdiv = np.std( v_prll )
    plt.text( 3.1e5, time_step_num,
              "mean={:.2e}".format( v_prll_mean ),
              fontsize = 10 )
    plt.text( 3.1e5, time_step_num - 0.3,
              "stdiv={:.2e}".format( v_prll_stdiv ),
              fontsize = 10 )
    #plt.plot( z_an, r_an_lower, color = "g" )
    h5file.close()


v_prll_an = [ v_prll_initial + a_prll * t for t in yticks_label ]
print( v_prll_an )
#print( a_prll )
plt.plot( v_prll_an, yticks, label = "theory", color = "b" )
    
plt.legend()
plt.yticks( yticks )
plt.yticks( yticks, yticks_label )
plt.savefig( "ion_velocities_as_points_prll.png" )


### V_perp

plt.figure()
plt.xlabel( "V_perp [cm/s]" )
plt.ylabel( "Time" )
yticks = range( len(out_files) )
yticks_label = []

for time_step_num, filename in enumerate(out_files):
    h5file = h5py.File( filename, mode = "r" )
    p_x_num = h5file["/Particle_sources/ions/momentum_x"]
    p_y_num = h5file["/Particle_sources/ions/momentum_y"]
    v_perp = np.sqrt( np.square( p_x_num ) + np.square( p_y_num ) ) / m

    time_step = h5file["/Time_grid"].attrs["current_time"][0]
    yticks_label.append( time_step )
    
    y = np.full_like( v_perp, time_step_num )
    plt.plot( v_perp, y, marker = '.', markersize = 10, color = "g", alpha = 0.005 )
    v_perp_mean = np.mean( v_perp )
    v_perp_stdiv = np.std( v_perp )
    plt.text( 3.1e5, time_step_num,
              "mean={:.2e}".format( v_perp_mean ),
              fontsize = 10 )
    plt.text( 3.1e5, time_step_num - 0.3,
              "stdiv={:.2e}".format( v_perp_stdiv ),
              fontsize = 10 )
    #plt.plot( z_an, r_an_upper, label = "theory", color = "g" )
    #plt.plot( z_an, r_an_lower, color = "g" )
    h5file.close()

#plt.legend()
plt.yticks( yticks )
plt.yticks( yticks, yticks_label )
plt.savefig( "ion_velocities_as_points_perp.png" )



### Plot as histogram

filename = 'thermolization_0000100.h5'
h5file = h5py.File( filename, mode = "r" )

p_x_num = h5file["/Particle_sources/ions/momentum_x"]
p_y_num = h5file["/Particle_sources/ions/momentum_y"]
p_z_num = h5file["/Particle_sources/ions/momentum_z"]
v_prll = p_z_num / m
v_perp = np.sqrt( np.square( p_x_num ) + np.square( p_y_num ) ) / m

plt.figure()
plt.xlabel( "V_prll [cm/s]" )
plt.ylabel( "N of Ions" )
plt.hist( v_prll )
#plt.ylim( 0, x_volume_size )
#plt.xlim( 0, z_volume_size )
#plt.plot( z_num, r_num, '.', label = "num" )
#plt.plot( z_an, r_an_upper, label = "theory", color = "g" )
#plt.plot( z_an, r_an_lower, color = "g" )
#plt.legend()
plt.savefig( "ion_velocities_hist.png" )
h5file.close()