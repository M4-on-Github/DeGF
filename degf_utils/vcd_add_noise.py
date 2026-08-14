"""Forward diffusion noising, used to build the contrastive "distorted" view.

Part of the VCD / DeGF decoding pipeline. The caller runs the model twice —
once on the clean image, once on a noised copy from here — and contrasts the
two token distributions. Detail the model asserts about the noised image, which
carries less real visual evidence, is treated as ungrounded.

An identical copy lives in DeGF/degf_utils/ and ONLY/only_utils/.
BenchyBench/tests/test_diffusion_noise.py runs the same contract against both
and asserts they stay byte-identical.

CAUTION: this is paper-method code. Its numerical behaviour determines
published results, so changes here are not routine refactoring. The tests pin
shape, determinism under a fixed seed, non-mutation of the input, and the
monotonicity of noise against the step — but they cannot tell you whether a
changed number is still the method described in the paper.
"""
import torch

def add_diffusion_noise(image_tensor, noise_step):
    """Apply forward diffusion noise at timestep `noise_step` (0-999).

    Implements the closed-form forward process, which reaches any timestep in
    one step rather than iterating:

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

    so `noise_step` interpolates between the original image (t=0, coefficient
    ~1) and near-pure Gaussian noise (t=999). The betas come from a sigmoid
    schedule over [-6, 6] rescaled to [1e-5, 5e-3].

    Args:
        image_tensor: image to noise. Any shape; batched input works, since
            every operation is elementwise.
        noise_step: timestep in [0, 999]. Cast with int(), so a float from a
            JSON config is accepted.

    Returns:
        A new tensor of the same shape and dtype. The input is NOT modified —
        the caller reuses the original for the clean forward pass, and mutating
        it in place would corrupt the contrastive comparison.

    Note:
        Uses the global torch RNG via randn_like, so reproducibility depends on
        the caller seeding torch. The run scripts do.

        The 1000-step schedule is rebuilt on every call. That is wasted work at
        one image per call, but it is left as-is deliberately: these tensors
        feed the published method, and the risk of changing a number outweighs
        the saving.
    """
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        """Sample x_t given x_0: retained signal plus scaled Gaussian noise."""
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step)

    return image_tensor_cd
