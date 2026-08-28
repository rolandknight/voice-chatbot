// TFLite Micro glue for microWakeWord. C++ because the TFLM API is C++ only.
// Exposed to wakeword.c through plain extern "C" functions.

#include <cstdint>
#include <cstddef>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// microWakeWord's streaming DS-CNN uses a small subset of TFLM ops. The
// resolver size (12) leaves headroom for future ops without paying for the
// AllOpsResolver footprint.
using MwwResolver = tflite::MicroMutableOpResolver<12>;

namespace {
MwwResolver make_resolver() {
    MwwResolver r;
    r.AddConv2D();
    r.AddDepthwiseConv2D();
    r.AddFullyConnected();
    r.AddReshape();
    r.AddSoftmax();
    r.AddQuantize();
    r.AddDequantize();
    r.AddAdd();
    r.AddMul();
    r.AddRelu();
    r.AddMean();
    r.AddLogistic();
    return r;
}
}  // namespace

struct MwwInterp {
    const tflite::Model        *model;
    MwwResolver                 resolver;
    tflite::MicroInterpreter   *interp;
};

extern "C" void *mww_create_interpreter(const uint8_t *model_bytes, size_t /*len*/,
                                        uint8_t *arena, size_t arena_size) {
    auto *m = new MwwInterp();
    m->model    = tflite::GetModel(model_bytes);
    m->resolver = make_resolver();
    m->interp   = new tflite::MicroInterpreter(m->model, m->resolver,
                                               arena, arena_size);
    if (m->interp->AllocateTensors() != kTfLiteOk) {
        delete m->interp;
        delete m;
        return nullptr;
    }
    return m;
}

extern "C" float mww_invoke(void *handle, const int16_t *frame, size_t samples) {
    auto *m = static_cast<MwwInterp *>(handle);
    TfLiteTensor *input = m->interp->input(0);

    // The streaming model takes one tile of mono int16 samples. The exact
    // tile size is baked into the model graph (matches `window_stride_ms`
    // in scripts/microwakeword/hey_*.yml). If the input tensor is smaller
    // than the incoming frame, we only feed the most-recent samples.
    size_t copy = static_cast<size_t>(input->bytes / sizeof(int16_t));
    if (copy > samples) copy = samples;
    const int16_t *src = frame + (samples - copy);
    int16_t *dst = reinterpret_cast<int16_t *>(input->data.raw);
    for (size_t i = 0; i < copy; i++) dst[i] = src[i];

    if (m->interp->Invoke() != kTfLiteOk) return 0.0f;

    TfLiteTensor *out = m->interp->output(0);
    // microWakeWord emits a single uint8 quantized probability in
    // [0..255]; convert to float.
    return static_cast<float>(out->data.uint8[0]) / 255.0f;
}
