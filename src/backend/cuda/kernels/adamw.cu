//
// zig-transformer-lab — AdamW step CUDA kernel (Stage 7, PR-mu)
//
// One kernel launch per parameter per step. The host loops over
// parameters, computes the bias-correction scalars bc1, bc2, and
// launches this kernel with the per-parameter pointers.
//
// Update rule (decoupled weight decay):
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g^2
//   m_hat = m * bc1       where bc1 = 1 / (1 - beta1^t)
//   v_hat = v * bc2
//   p -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
//
// All arrays are flat f32 of length N = totalElements(param.shape).
// params, m, v are updated in-place. grads is read-only.
//

extern "C" __global__
void adamw_step(
    float* params,
    const float* grads,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1,
    float bc2,
    unsigned int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float g = grads[i];
    float m_new = beta1 * m[i] + (1.0f - beta1) * g;
    float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_new;
    v[i] = v_new;

    float m_hat = m_new * bc1;
    float v_hat = v_new * bc2;

    float p = params[i];
    params[i] = p - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p);
}
