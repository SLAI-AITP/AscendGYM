# gather_pa_kv_cache Build Dependencies

## Current Bottleneck

### Problem
Sandbox build fails because **third-party dependencies cannot be downloaded**:
```
error: downloading 'https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/json/json-3.11.3.tar.gz' failed
gmake[2]: *** [CMakeFiles/third_party_json.dir/build.make:98: ...third_party_json-download] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:410: CMakeFiles/third_party_json.dir/all] Error 2
gmake: *** [Makefile:156: all] Error 2
```

### Why Earlier Builds Worked
CMake caches downloaded dependencies in the build directory. Once third-party libs (json, protobuf, eigen) were downloaded, subsequent builds reused the cached copies. The sandbox container was reset, clearing the cache and forcing re-download.

### Why This Is a Bottleneck
1. **Network dependency**: Build requires internet access to `cann-3rd.obs.cn-north-4.myhuaweicloud.com`
2. **SSL issues**: The sandbox proxy configuration (`http_proxy=http://127.0.0.1:3128`) causes SSL certificate verification failures
3. **No local cache**: CANN 9.1.0 is only available inside sandbox containers; local `cann-8.5.1` doesn't help
4. **No CANN-bundled fallback**: The ops-transformer CMake doesn't use CANN's bundled third-party libs by default

## Required Dependencies

### Build System Dependencies (downloaded by CMake)
These are downloaded during the build process and must be available:

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| **json** | 3.11.3 | `cann-3rd.obs.cn-north-4.myhuaweicloud.com/json/` | JSON parsing for tiling data |
| **protobuf** | 25.1 | Same CDN | Protocol buffer serialization |
| **eigen** | (bundled) | GitCode CDN | Matrix operations |
| **abseil-cpp** | (bundled) | GitCode CDN | Core C++ utilities |

### CANN Dependencies (must be installed)
| Component | Version | Purpose |
|-----------|---------|---------|
| **CANN Toolkit** | 9.1.0 | Core Ascend AI framework |
| **AscendC** | 9.1.0 | AI core kernel development |
| **torch_npu** | 9.1.0 | PyTorch Ascend integration |
| **driver** | 9.1.0 | Ascend hardware driver |

### Runtime Dependencies (Python)
| Package | Purpose |
|---------|---------|
| `torch` | PyTorch tensor operations |
| `torch_npu` | Ascend NPU backend |
| `gtsrusops` | Auto-generated Python bindings |

## How to Fix

### ✅ Option 1 (APPLIED): Pre-cache in Dockerfile
The Dockerfile (`sandbox/client/docker/Dockerfile`) now:
1. **Disables SSL verification** for git/curl globally:
   ```dockerfile
   ENV GIT_SSL_NO_VERIFY=true
   ENV GIT_CONFIG_KEY_0=http.sslVerify
   ENV GIT_CONFIG_VALUE_0=false
   RUN git config --global http.sslVerify false
   ```
2. **Pre-clones third-party repos** into `/sandboxfusion/third_party/`:
   ```dockerfile
   git clone --depth=1 https://gitcode.com/cann/ops-tensor.git /sandboxfusion/third_party/ops-tensor
   git clone --depth=1 https://gitcode.com/cann/opbase.git /sandboxfusion/third_party/opbase
   ```
3. The ops-transformer CMake system checks `${PROJECT_SOURCE_DIR}/third_party/` first.
   If found locally, it **skips the network download entirely**.

### Option 2: Use CANN-bundled third-party libs
The CANN package includes pre-built third-party libraries. Configure CMake to use them:
```bash
export CANN_3RD_LIB_PATH=/usr/local/Ascend/cann-9.1.0/third_party/lib_cache
```

### Option 3: Fix proxy configuration
Ensure the sandbox proxy handles HTTPS properly:
```bash
export https_proxy=http://127.0.0.1:3128
export no_proxy=127.0.0.*,localhost,*.huawei.com,10.*,<NODE_IP>,7.212.104.213,127.0.0.1
```

## Build Process Summary

The ops-transformer build process:
1. Download third-party dependencies (json, protobuf, eigen, abseil)
2. Build tiling host library (`liboptiling.so`)
3. Compile kernel source files (AscendC)
4. Package into `.run` installer
5. Install custom OPP
6. Generate Python bindings
7. Install `gtsrusops` package

**Current blocker**: Step 1 fails due to network/SSL issues.

## Kernel Files Structure

```
gather_pa_kv_cache/
├── op_host/                          # Host-side code
│   ├── gather_pa_kv_cache_def.cpp   # OpDef registration
│   ├── gather_pa_kv_cache_tiling.cpp      # Tiling for old arch
│   ├── gather_pa_kv_cache_tiling_arch35.cpp  # Tiling for arch35 (CANN 9.1.0)
│   └── gather_pa_kv_cache_tiling.h        # Tiling data structures
├── op_kernel/                        # Kernel source
│   ├── gather_pa_kv_cache.cpp       # Old arch kernel (keys 618/619/621/577)
│   ├── gather_pa_kv_cache_apt.cpp   # APT kernel (keys 1000-1111)
│   ├── gather_pa_kv_cache_nd.h      # ND format implementation
│   ├── gather_pa_kv_cache_nz.h      # NZ format implementation
│   └── arch35/                      # Arch35-specific headers
│       ├── gather_pa_kv_cache_nd.h
│       └── gather_pa_kv_cache_nz.h
└── CMakeLists.txt                   # Build configuration
```

## Dependencies vs Kernel Architecture

| Component | ascend910b | ascend950 |
|-----------|------------|-----------|
| **Tiling keys** | 618/619/621/577 (old) | 1000-1111 (arch35) |
| **Kernel file** | `gather_pa_kv_cache.cpp` | `gather_pa_kv_cache_apt.cpp` |
| **Tiling source** | `gather_pa_kv_cache_tiling.cpp` | `gather_pa_kv_cache_tiling_arch35.cpp` |
| **Third-party deps** | Same (json, protobuf, eigen) | Same |

## Third-Party Dependency Details

### json (3.11.3)
- **Download URL**: `https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/json/json-3.11.3.tar.gz`
- **Used by**: Tiling data serialization
- **Fallback**: CANN may have bundled version at `/usr/local/Ascend/cann-9.1.0/third_party/lib_cache/json/`

### protobuf (25.1)
- **Download URL**: `https://gitcode.com/cann/protobuf.git`
- **Used by**: Protocol buffer serialization for tiling data
- **Fallback**: CANN bundled at `/usr/local/Ascend/cann-9.1.0/lib64/libprotoc.a`

### eigen
- **Download URL**: `https://gitcode.com/cann/ops-tensor.git`
- **Used by**: Matrix operations in tiling
- **Fallback**: May be bundled with CANN

### abseil-cpp
- **Download URL**: `https://gitcode.com/cann/abseil-cpp.git`
- **Used by**: Core C++ utilities
- **Fallback**: CANN may have bundled version

## Notes

1. The actual dependency list is determined by the CMake `find_package()` calls in the build system
2. CANN 9.1.0 should include bundled third-party libs, but the CMake configuration doesn't use them by default
3. Once third-party deps are available, kernel compilation itself is fast (~2 minutes)
4. The third-party download is the **only** external dependency that blocks the build
