# Profiling

The `matrix multiplication` is often used to profile the performance of a CPU and the system.

To get the best performance, first of all configure `X-HEEP` as following:

`make mcu-gen X_HEEP_CFG=configs/benchmark.hjson`

This configuration contains the best settings for `X-HEEP` to compute a fast matrix multiplication.

## Standard RISC-V

To compile the application with a standard `RISC-V` compiler and `ISA`:

`make app PROJECT=example_matmul COMPILER_PREFIX=riscv32-unknown- ARCH=rv32imc`

This will compile the code using 8-bit input data. While if you want to use 32- or 16-bit input data:

`make app PROJECT=example_matmul COMPILER_PREFIX=riscv32-unknown- ARCH=rv32imc COMPILER_FLAGS="-DMATMUL32"` 

or `COMPILER_FLAGS="-DMATMUL16"`

## CORE-V RISC-V ISA Extensions

`make app PROJECT=example_matmul COMPILER_PREFIX=riscv32-corev- ARCH=rv32imc_zicsr_zifencei_xcvhwlp_xcvmem_xcvmac_xcvbi_xcvalu_xcvsimd_xcvbitmanip COMPILER_FLAGS="-DMATMUL32 -D__PULP_EXTENSIONS"` 

C