TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
    config.cpp \
    domain.cpp \
    External_field.cpp \
    field_solver.cpp \
    inner_region.cpp \
    parse_cmd_line.cpp \
    particle.cpp \
    particle_interaction_model.cpp \
    particle_source.cpp \
    particle_to_mesh_map.cpp \
    spatial_mesh.cpp \
    time_grid.cpp \
    vec3d.cpp \
    lib/tinyexpr/tinyexpr.c

HEADERS += \
    config.h \
    domain.h \
    External_field.h \
    field_solver.h \
    general_kernels.h \
    inner_region.h \
    node_reference.h \
    parse_cmd_line.h \
    particle.h \
    particle_interaction_model.h \
    particle_source.h \
    particle_to_mesh_map.h \
    physical_constants.h \
    spatial_mesh.h \
    time_grid.h \
    vec3d.h \
    lib/tinyexpr/tinyexpr.h

CUDA_SOURCES += \
    general_kernels.cu

INCLUDEPATH += "c:/Program Files/HDF_Group/HDF5/1.10.5/include"
INCLUDEPATH += "c:/boost_1_70_0"

QMAKE_LIBDIR += "c:/boost_1_70_0/lib64-msvc-14.0"
QMAKE_LIBDIR += "c:/Program Files/HDF_Group/HDF5/1.10.5/lib"

LIBS += -llibhdf5 -llibhdf5_cpp -llibhdf5_hl -llibhdf5_hl_cpp -llibzlib -llibszip

CUDA_DIR = "c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1"
SYSTEM_NAME = x64
SYSTEM_TYPE = 64
GENCODE1 = arch=compute_61,code=sm_61
NVCC_OPTIONS = --use_fast_math

INCLUDEPATH += $$CUDA_DIR/include

QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME

LIBS += -lcuda -lcudart -lcublas -lcurand

CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

CONFIG(debug, debug|release) {
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = debug/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine $$SYSTEM_TYPE -Xcompiler "/MDd" -gencode $$GENCODE1 -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    cuda.input = CUDA_SOURCES
    cuda.output = release/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine $$SYSTEM_TYPE -Xcompiler "/MD" -gencode $$GENCODE1 -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}


