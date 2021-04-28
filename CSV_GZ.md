# Do once per login:

```bash
srun -N 1 -t 480:0 --pty bash
module load cuda/11.0.1
conda activate cudf_dev1
```

# To rebuild:

```bash
cd /gpfs/fs1/esoha/cudf/cpp/build
make -j 50
make install
```

# To test:

```bash
python -c "import cudf; print(len(cudf.read_csv(\"/gpfs/fs1/esoha/multi_transcoding/part-00000-d640ace7-b38b-45c2-917e-3264c51dd555-c000.csv.gz\")))"
```

# Notes:

The code that does the CPUdeflate is here:
https://github.com/rapidsai/cudf/blob/348ad4dcdd1cc7f970241d4389493176cd58dd5a/cpp/src/io/csv/reader_impl.cu#L225

Test code that does GPU inflate: https://github.com/rapidsai/cudf/blob/348ad4dcdd1cc7f970241d4389493176cd58dd5a/cpp/tests/io/comp/decomp_test.cu

Input structure for the inflate is `cudf::io::gpu_inflate_input_s`:
https://github.com/rapidsai/cudf/blob/348ad4dcdd1cc7f970241d4389493176cd58dd5a/cpp/src/io/comp/gpuinflate.h#L28

Output structure for the inflate is `gpu_inflate_status_s`:
https://github.com/rapidsai/cudf/blob/348ad4dcdd1cc7f970241d4389493176cd58dd5a/cpp/src/io/comp/gpuinflate.h#L38

To use deflate, copy those structs into GPU memory and then call GPU defalte.  To allocate those structures in GPU memory, write them like this:

```cpp
  rmm::device_vector<cudf::io::gpu_inflate_input_s> d_inf_args;
  rmm::device_vector<cudf::io::gpu_inflate_status_s> d_inf_stat;
```

rfc1952 describes the header format for a .gz file, which can include a filename, comments, CRC, compression type, etc.  Of course, it also includes the compressed blocks.  The compression type is usually 8, which is deflate, which is described by rfc1951.

gpuinflate test code:
https://github.com/rapidsai/cudf/blob/348ad4dcdd1cc7f970241d4389493176cd58dd5a/cpp/tests/io/comp/decomp_test.cu

The gzip headers are supported by `gpuinflate` and they are ignored.

