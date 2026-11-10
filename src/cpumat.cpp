// FP32 row-major GEMM for Apple Silicon. Compiled without fast-math.
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
constexpr int BM = 128, BN = 128, BK = 256, MR = 4, NR = 8;

void message(char* out, size_t size, const char* text) noexcept {
    if (out && size) std::snprintf(out, size, "%s", text ? text : "unknown error");
}
size_t product(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::overflow_error("matrix size overflow");
    return a * b;
}
size_t float_count(size_t a, size_t b) {
    size_t count = product(a, b);
    product(count, sizeof(float));
    return count;
}

float load_scalar(const float* base, size_t index) noexcept {
    float value;
    std::memcpy(&value, reinterpret_cast<const unsigned char*>(base) + index * sizeof(float), sizeof(value));
    return value;
}
struct Config {

    int bm, bn, bk, mr, nr;
    int requested_threads, actual_threads, packed_b;
};

struct Plan;
struct Context {
    int requested_threads, capacity;
    std::vector<std::thread> workers;
    std::mutex mutex, run_mutex;
    std::condition_variable start, done;
    size_t generation = 0;
    int remaining = 0, active = 0;
    bool stopping = false;
    Plan* job = nullptr;

    explicit Context(int threads);
    ~Context();
    void shutdown() noexcept;
    void worker(int id) noexcept;
    void run(Plan* plan);
};

struct Plan {
    int implementation, m, k, n;
    const float *a, *b;
    Config config;
    std::unique_ptr<float[]> output;
    std::unique_ptr<float[]> packed;
    Context* context;
    std::mutex run_mutex;
    bool initialized = false;

    Plan(Context* context_, int impl, int m_, int k_, int n_, const float* a_, const float* b_)
        : implementation(impl), m(m_), k(k_), n(n_), a(a_), b(b_),
          config{BM, BN, BK, MR, NR, context_->requested_threads, 1, impl == 0},
          output(new float[float_count(static_cast<size_t>(m_), static_cast<size_t>(n_))]),
          context(context_) {
        if (impl != 0) return;
        size_t panels = (static_cast<size_t>(n) + NR - 1) / NR;
        packed.reset(new float[float_count(product(panels, static_cast<size_t>(k)), NR)]);
        for (size_t panel = 0; panel < panels; ++panel)
            for (int p = 0; p < k; ++p)
                for (int lane = 0; lane < NR; ++lane) {
                    size_t col = panel * NR + static_cast<size_t>(lane);
                    packed[(panel * static_cast<size_t>(k) + static_cast<size_t>(p)) * NR + lane] =
                        col < static_cast<size_t>(n) ? load_scalar(b, static_cast<size_t>(p) * n + col) : 0.0f;
                }
        const size_t row_blocks = (static_cast<size_t>(m) + BM - 1) / BM;
        config.actual_threads = static_cast<int>(std::min(row_blocks, static_cast<size_t>(context->capacity)));
    }

    template<int Rows>
    void microkernel(size_t row, size_t col, size_t p0, size_t pend) noexcept {
        float32x4_t lo[Rows], hi[Rows];
        const size_t width = std::min(static_cast<size_t>(NR), static_cast<size_t>(n) - col);
        for (int r = 0; r < Rows; ++r) {
            float* c = output.get() + (row + r) * static_cast<size_t>(n) + col;
            if (p0 == 0) {
                lo[r] = vdupq_n_f32(0); hi[r] = vdupq_n_f32(0);
            } else if (width == NR) {
                lo[r] = vld1q_f32(c); hi[r] = vld1q_f32(c + 4);
            } else {
                float tmp[NR] = {};
                for (size_t j = 0; j < width; ++j) tmp[j] = c[j];
                lo[r] = vld1q_f32(tmp); hi[r] = vld1q_f32(tmp + 4);
            }
        }
        const float* bp = packed.get() + ((col / NR) * static_cast<size_t>(k) + p0) * NR;
        for (size_t p = p0; p < pend; ++p, bp += NR) {
            float32x4_t b0 = vld1q_f32(bp), b1 = vld1q_f32(bp + 4);
            for (int r = 0; r < Rows; ++r) {
                float av = load_scalar(a, (row + r) * static_cast<size_t>(k) + p);
                lo[r] = vfmaq_n_f32(lo[r], b0, av);
                hi[r] = vfmaq_n_f32(hi[r], b1, av);
            }
        }
        for (int r = 0; r < Rows; ++r) {
            float* c = output.get() + (row + r) * static_cast<size_t>(n) + col;
            if (width == NR) {
                vst1q_f32(c, lo[r]); vst1q_f32(c + 4, hi[r]);
            } else {
                float tmp[NR];
                vst1q_f32(tmp, lo[r]); vst1q_f32(tmp + 4, hi[r]);
                for (size_t j = 0; j < width; ++j) c[j] = tmp[j];
            }
        }
    }
    void compute(int id) noexcept {
        const size_t mm = static_cast<size_t>(m), nn = static_cast<size_t>(n), kk = static_cast<size_t>(k);
        const size_t blocks = (mm + BM - 1) / BM;
        for (size_t block = static_cast<size_t>(id); block < blocks; block += config.actual_threads) {
            size_t rb = block * BM, re = std::min(rb + BM, mm);
            for (size_t nb = 0; nb < nn; nb += BN) {
                size_t ne = std::min(nb + BN, nn);
                for (size_t kb = 0; kb < kk; kb += BK) {
                    size_t ke = std::min(kb + BK, kk);
                    for (size_t i = rb; i < re; i += MR)
                        for (size_t j = nb; j < ne; j += NR) {
                            switch (std::min(static_cast<size_t>(MR), re - i)) {
                                case 4: microkernel<4>(i, j, kb, ke); break;
                                case 3: microkernel<3>(i, j, kb, ke); break;
                                case 2: microkernel<2>(i, j, kb, ke); break;
                                default: microkernel<1>(i, j, kb, ke); break;
                            }
                        }
                }
            }
        }
    }
    void run() {
        std::lock_guard<std::mutex> serial(run_mutex);
        if (implementation == 1) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                        a, k, b, n, 0.0f, output.get(), n);
        } else if (config.actual_threads == 1) {
            compute(0);
        } else {
            context->run(this);
        }
        initialized = true;
    }
};

Context::Context(int threads)
    : requested_threads(threads),
      capacity(static_cast<int>(std::min(static_cast<unsigned>(threads),
                                          std::max(1u, std::thread::hardware_concurrency())))) {
    try {
        workers.reserve(static_cast<size_t>(capacity - 1));
        for (int id = 1; id < capacity; ++id) workers.emplace_back(&Context::worker, this, id);
    } catch (...) {
        shutdown();
        throw;
    }
}
Context::~Context() { shutdown(); }
void Context::shutdown() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    start.notify_all();
    for (auto& worker : workers) if (worker.joinable()) worker.join();
    workers.clear();
}
void Context::worker(int id) noexcept {
    size_t seen = 0;
    std::unique_lock<std::mutex> lock(mutex);
    for (;;) {
        start.wait(lock, [&] { return stopping || generation != seen; });
        if (stopping) return;
        seen = generation;
        if (id >= active) continue;
        Plan* current = job;
        lock.unlock();
        current->compute(id);
        lock.lock();
        if (--remaining == 0) done.notify_one();
    }
}
void Context::run(Plan* plan) {
    std::lock_guard<std::mutex> serial(run_mutex);
    {
        std::lock_guard<std::mutex> lock(mutex);
        job = plan;
        active = plan->config.actual_threads;
        remaining = active - 1;
        ++generation;
    }
    start.notify_all();
    plan->compute(0);
    std::unique_lock<std::mutex> lock(mutex);
    done.wait(lock, [&] { return remaining == 0; });
}
}

extern "C" {
int cpumat_abi_version() noexcept { return 1; }
void* cpumat_create(int threads, char* error, size_t error_size) noexcept {
    try {
        if (threads <= 0) throw std::invalid_argument("threads must be positive");
        return new Context(threads);
    } catch (const std::exception& e) { message(error, error_size, e.what()); }
      catch (...) { message(error, error_size, "unknown native exception"); }
    return nullptr;
}
void cpumat_destroy_context(void* context) noexcept { delete static_cast<Context*>(context); }
void* cpumat_prepare(void* context, int impl, int m, int k, int n, const float* a, const float* b,
                     char* error, size_t error_size) noexcept {
    try {
        if (!context || (impl != 0 && impl != 1) || m <= 0 || k <= 0 || n <= 0 || !a || !b)
            throw std::invalid_argument("invalid CPU GEMM arguments");
        float_count(static_cast<size_t>(m), static_cast<size_t>(k));
        float_count(static_cast<size_t>(k), static_cast<size_t>(n));
        return new Plan(static_cast<Context*>(context), impl, m, k, n, a, b);
    } catch (const std::exception& e) { message(error, error_size, e.what()); }
      catch (...) { message(error, error_size, "unknown native exception"); }
    return nullptr;
}
int cpumat_run(void* handle, char* error, size_t error_size) noexcept {
    try {
        if (!handle) throw std::invalid_argument("null CPU GEMM plan");
        static_cast<Plan*>(handle)->run();
        return 0;
    } catch (const std::exception& e) { message(error, error_size, e.what()); }
      catch (...) { message(error, error_size, "unknown native exception"); }
    return -1;
}
int cpumat_config(void* handle, Config* config, char* error, size_t error_size) noexcept {
    if (!handle || !config) { message(error, error_size, "null CPU GEMM plan or config"); return -1; }
    *config = static_cast<Plan*>(handle)->config;
    return 0;
}
const float* cpumat_output(void* handle) noexcept {
    return handle && static_cast<Plan*>(handle)->initialized ? static_cast<Plan*>(handle)->output.get() : nullptr;
}
void cpumat_destroy(void* handle) noexcept { delete static_cast<Plan*>(handle); }
}
