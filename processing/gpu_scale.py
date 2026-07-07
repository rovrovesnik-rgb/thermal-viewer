"""
GPU-accelerated image scaling using OpenCL.
Supports AMD GPUs via ROCm/OpenCL.
"""

import numpy as np
from typing import Optional, Tuple
from enum import Enum, auto
import cv2


class ScaleAlgorithm(Enum):
    """Available scaling algorithms."""
    NEAREST = auto()    # Fastest, pixelated
    BILINEAR = auto()   # Balanced quality/speed
    LANCZOS = auto()    # High quality, slower
    CUBIC = auto()      # Bicubic interpolation
    SHARP = auto()      # Lanczos + edge enhancement
    XBRZ = auto()       # Edge-aware pixel art style (crisp)


# Try to import OpenCL
_HAS_OPENCL = False
_OCL_CONTEXT = None
_OCL_QUEUE = None
_OCL_PROGRAM = None

try:
    import pyopencl as cl
    _HAS_OPENCL = True
except ImportError:
    pass


# OpenCL kernel source for scaling
_OCL_KERNEL_SOURCE = """
__kernel void scale_nearest(
    __global const uchar* src,
    __global uchar* dst,
    const int src_width,
    const int src_height,
    const int dst_width,
    const int dst_height,
    const int channels
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= dst_width || y >= dst_height) return;

    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;

    int src_x = (int)(x * scale_x);
    int src_y = (int)(y * scale_y);

    src_x = min(src_x, src_width - 1);
    src_y = min(src_y, src_height - 1);

    int src_idx = (src_y * src_width + src_x) * channels;
    int dst_idx = (y * dst_width + x) * channels;

    for (int c = 0; c < channels; c++) {
        dst[dst_idx + c] = src[src_idx + c];
    }
}

__kernel void scale_bilinear(
    __global const uchar* src,
    __global uchar* dst,
    const int src_width,
    const int src_height,
    const int dst_width,
    const int dst_height,
    const int channels
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= dst_width || y >= dst_height) return;

    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;

    float src_xf = x * scale_x;
    float src_yf = y * scale_y;

    int x0 = (int)src_xf;
    int y0 = (int)src_yf;
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    float dx = src_xf - x0;
    float dy = src_yf - y0;

    int dst_idx = (y * dst_width + x) * channels;

    for (int c = 0; c < channels; c++) {
        float v00 = src[(y0 * src_width + x0) * channels + c];
        float v10 = src[(y0 * src_width + x1) * channels + c];
        float v01 = src[(y1 * src_width + x0) * channels + c];
        float v11 = src[(y1 * src_width + x1) * channels + c];

        float v0 = v00 * (1 - dx) + v10 * dx;
        float v1 = v01 * (1 - dx) + v11 * dx;
        float v = v0 * (1 - dy) + v1 * dy;

        dst[dst_idx + c] = (uchar)clamp(v, 0.0f, 255.0f);
    }
}

// Lanczos kernel function
float lanczos(float x, float a) {
    if (x == 0) return 1.0f;
    if (fabs(x) >= a) return 0.0f;
    float pi_x = 3.14159265359f * x;
    return (a * sin(pi_x) * sin(pi_x / a)) / (pi_x * pi_x);
}

__kernel void scale_lanczos(
    __global const uchar* src,
    __global uchar* dst,
    const int src_width,
    const int src_height,
    const int dst_width,
    const int dst_height,
    const int channels
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= dst_width || y >= dst_height) return;

    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;

    float src_xf = x * scale_x;
    float src_yf = y * scale_y;

    int dst_idx = (y * dst_width + x) * channels;
    const float a = 3.0f;  // Lanczos-3

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        int x_start = max((int)(src_xf - a), 0);
        int x_end = min((int)(src_xf + a) + 1, src_width);
        int y_start = max((int)(src_yf - a), 0);
        int y_end = min((int)(src_yf + a) + 1, src_height);

        for (int sy = y_start; sy < y_end; sy++) {
            float wy = lanczos(src_yf - sy, a);
            for (int sx = x_start; sx < x_end; sx++) {
                float wx = lanczos(src_xf - sx, a);
                float w = wx * wy;
                sum += src[(sy * src_width + sx) * channels + c] * w;
                weight_sum += w;
            }
        }

        if (weight_sum > 0) {
            dst[dst_idx + c] = (uchar)clamp(sum / weight_sum, 0.0f, 255.0f);
        }
    }
}
"""


def _init_opencl() -> bool:
    """Initialize OpenCL context and compile kernels."""
    global _OCL_CONTEXT, _OCL_QUEUE, _OCL_PROGRAM

    if not _HAS_OPENCL:
        print("OpenCL: pyopencl not installed")
        return False

    try:
        # Get platforms
        platforms = cl.get_platforms()
        if not platforms:
            print("OpenCL: No platforms found")
            print("Note: For AMD GPUs, ensure RUSTICL_ENABLE=radeonsi is set")
            return False

        print(f"OpenCL: Found {len(platforms)} platform(s)")

        # Find GPU device - prefer AMD
        gpu_device = None
        for i, platform in enumerate(platforms):
            print(f"  Platform {i}: {platform.name}")
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                for device in devices:
                    print(f"    GPU: {device.name} ({device.vendor})")
                    # Prefer AMD devices (gfx1030 = Radeon 6950XT RDNA2)
                    if any(x in device.vendor.lower() for x in ['amd', 'advanced micro']):
                        gpu_device = device
                        break
                    elif gpu_device is None:
                        gpu_device = device
            except cl.RuntimeError as e:
                print(f"    (No GPU devices: {e})")

        if gpu_device is None:
            print("OpenCL: No GPU device found")
            print("Note: For AMD GPUs on Linux, run with: RUSTICL_ENABLE=radeonsi python ...")
            return False

        # Create context and queue
        _OCL_CONTEXT = cl.Context([gpu_device])
        _OCL_QUEUE = cl.CommandQueue(_OCL_CONTEXT)

        # Compile kernels
        _OCL_PROGRAM = cl.Program(_OCL_CONTEXT, _OCL_KERNEL_SOURCE).build()

        # Map GPU codenames to marketing names
        gpu_name = gpu_device.name
        gpu_friendly = gpu_name
        codename_map = {
            'gfx1030': 'Radeon RX 6900/6950 XT',
            'gfx1031': 'Radeon RX 6800/6800 XT',
            'gfx1032': 'Radeon RX 6700 XT',
            'gfx1034': 'Radeon RX 6600 XT',
            'gfx1100': 'Radeon RX 7900 XTX',
            'gfx1101': 'Radeon RX 7900 XT',
            'gfx1102': 'Radeon RX 7800/7700',
        }
        for code, name in codename_map.items():
            if code in gpu_name.lower():
                gpu_friendly = f"{name} ({gpu_name})"
                break

        print(f"OpenCL initialized: {gpu_friendly}")
        return True

    except Exception as e:
        print(f"OpenCL initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


class GPUScaler:
    """
    GPU-accelerated image scaler with multiple algorithm support.
    Falls back to CPU (OpenCV) if GPU is unavailable.
    """

    def __init__(self):
        self._gpu_enabled = False
        self._algorithm = ScaleAlgorithm.SHARP  # Default to sharp for best quality
        self._device_name = "CPU"

        # Try to initialize OpenCL
        if _HAS_OPENCL and _init_opencl():
            self._gpu_enabled = True
            self._device_name = self._get_device_name()

    def _get_device_name(self) -> str:
        """Get the GPU device name (with friendly marketing name if possible)."""
        if not _HAS_OPENCL or _OCL_CONTEXT is None:
            return "CPU"

        try:
            devices = _OCL_CONTEXT.get_info(cl.context_info.DEVICES)
            if devices:
                gpu_name = devices[0].name
                # Map GPU codenames to marketing names
                codename_map = {
                    'gfx1030': 'RX 6950XT',
                    'gfx1031': 'RX 6800XT',
                    'gfx1032': 'RX 6700XT',
                    'gfx1034': 'RX 6600XT',
                    'gfx1100': 'RX 7900XTX',
                    'gfx1101': 'RX 7900XT',
                    'gfx1102': 'RX 7800XT',
                }
                for code, name in codename_map.items():
                    if code in gpu_name.lower():
                        return name
                return gpu_name
        except Exception:
            pass
        return "GPU"

    @property
    def gpu_available(self) -> bool:
        """Check if GPU scaling is available."""
        return _HAS_OPENCL and _OCL_CONTEXT is not None

    @property
    def gpu_enabled(self) -> bool:
        """Check if GPU scaling is currently enabled."""
        return self._gpu_enabled and self.gpu_available

    @gpu_enabled.setter
    def gpu_enabled(self, value: bool):
        """Enable or disable GPU scaling."""
        self._gpu_enabled = value and self.gpu_available

    @property
    def algorithm(self) -> ScaleAlgorithm:
        """Get current scaling algorithm."""
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: ScaleAlgorithm):
        """Set scaling algorithm (hot-swappable, no restart needed)."""
        self._algorithm = value

    @property
    def device_name(self) -> str:
        """Get the current processing device name."""
        if self.gpu_enabled:
            return self._device_name
        return "CPU"

    def scale(
        self,
        image: np.ndarray,
        scale_factor: float = 2.0,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Scale an image using the configured method.

        Args:
            image: Input image (grayscale or BGR)
            scale_factor: Scale multiplier (ignored if target_size is set)
            target_size: Target (width, height) tuple

        Returns:
            Scaled image
        """
        if target_size:
            dst_width, dst_height = target_size
        else:
            dst_height = int(image.shape[0] * scale_factor)
            dst_width = int(image.shape[1] * scale_factor)

        if self.gpu_enabled:
            return self._scale_gpu(image, dst_width, dst_height)
        else:
            return self._scale_cpu(image, dst_width, dst_height)

    def _scale_gpu(self, image: np.ndarray, dst_width: int, dst_height: int) -> np.ndarray:
        """Scale using OpenCL GPU."""
        src_height, src_width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]

        # Ensure contiguous array
        src = np.ascontiguousarray(image)

        # Create output buffer
        if image.ndim == 2:
            dst = np.zeros((dst_height, dst_width), dtype=np.uint8)
        else:
            dst = np.zeros((dst_height, dst_width, channels), dtype=np.uint8)

        try:
            # Create OpenCL buffers
            mf = cl.mem_flags
            src_buf = cl.Buffer(_OCL_CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src)
            dst_buf = cl.Buffer(_OCL_CONTEXT, mf.WRITE_ONLY, dst.nbytes)

            # Select kernel
            if self._algorithm == ScaleAlgorithm.NEAREST:
                kernel = _OCL_PROGRAM.scale_nearest
            elif self._algorithm == ScaleAlgorithm.LANCZOS:
                kernel = _OCL_PROGRAM.scale_lanczos
            else:  # BILINEAR or CUBIC (use bilinear for now)
                kernel = _OCL_PROGRAM.scale_bilinear

            # Execute kernel
            kernel(
                _OCL_QUEUE,
                (dst_width, dst_height),
                None,
                src_buf, dst_buf,
                np.int32(src_width), np.int32(src_height),
                np.int32(dst_width), np.int32(dst_height),
                np.int32(channels)
            )

            # Read result
            cl.enqueue_copy(_OCL_QUEUE, dst, dst_buf).wait()

            return dst

        except Exception as e:
            print(f"GPU scaling failed, falling back to CPU: {e}")
            return self._scale_cpu(image, dst_width, dst_height)

    def _scale_cpu(self, image: np.ndarray, dst_width: int, dst_height: int) -> np.ndarray:
        """Scale using OpenCV CPU."""
        if self._algorithm == ScaleAlgorithm.SHARP:
            return self._scale_sharp(image, dst_width, dst_height)
        elif self._algorithm == ScaleAlgorithm.XBRZ:
            return self._scale_xbrz(image, dst_width, dst_height)
        else:
            interpolation = self._get_cv2_interpolation()
            return cv2.resize(image, (dst_width, dst_height), interpolation=interpolation)

    def _scale_sharp(self, image: np.ndarray, dst_width: int, dst_height: int) -> np.ndarray:
        """
        Scale with edge-aware sharpening.
        Uses Lanczos upscale + unsharp mask for crisp edges.
        """
        # First upscale with Lanczos
        scaled = cv2.resize(image, (dst_width, dst_height), interpolation=cv2.INTER_LANCZOS4)

        # Apply unsharp mask for edge enhancement
        # Gaussian blur
        blurred = cv2.GaussianBlur(scaled, (0, 0), 1.5)

        # Unsharp mask: original + (original - blurred) * amount
        # amount = 0.7 for moderate sharpening
        sharpened = cv2.addWeighted(scaled, 1.7, blurred, -0.7, 0)

        return sharpened

    def _scale_xbrz(self, image: np.ndarray, dst_width: int, dst_height: int) -> np.ndarray:
        """
        Edge-aware scaling inspired by xBR/xBRZ algorithms.
        Preserves edges while smoothing flat regions.
        """
        src_height, src_width = image.shape[:2]
        scale_x = dst_width / src_width
        scale_y = dst_height / src_height

        # For thermal images, use edge-directed interpolation approach:
        # 1. Detect edges in source
        # 2. Use nearest-neighbor near edges, bilinear in smooth areas

        # Convert to grayscale for edge detection if color
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect edges with Sobel
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_mag = (edge_mag / edge_mag.max() * 255).astype(np.uint8) if edge_mag.max() > 0 else edge_mag.astype(np.uint8)

        # Scale edge map
        edge_scaled = cv2.resize(edge_mag, (dst_width, dst_height), interpolation=cv2.INTER_LINEAR)

        # Create two versions: sharp (Lanczos) and smooth (area/cubic)
        sharp_scaled = cv2.resize(image, (dst_width, dst_height), interpolation=cv2.INTER_LANCZOS4)
        smooth_scaled = cv2.resize(image, (dst_width, dst_height), interpolation=cv2.INTER_CUBIC)

        # Blend based on edge strength - more edge = more sharp version
        edge_weight = edge_scaled.astype(np.float32) / 255.0

        # Apply sharpening to the sharp version
        blurred = cv2.GaussianBlur(sharp_scaled, (0, 0), 1.0)
        sharp_enhanced = cv2.addWeighted(sharp_scaled, 1.5, blurred, -0.5, 0)

        if image.ndim == 3:
            edge_weight = edge_weight[:, :, np.newaxis]

        # Blend: use sharp_enhanced where edges are strong, smooth elsewhere
        result = (sharp_enhanced * edge_weight + smooth_scaled * (1 - edge_weight)).astype(np.uint8)

        return result

    def _get_cv2_interpolation(self) -> int:
        """Get OpenCV interpolation constant for current algorithm."""
        if self._algorithm == ScaleAlgorithm.NEAREST:
            return cv2.INTER_NEAREST
        elif self._algorithm == ScaleAlgorithm.BILINEAR:
            return cv2.INTER_LINEAR
        elif self._algorithm == ScaleAlgorithm.LANCZOS:
            return cv2.INTER_LANCZOS4
        elif self._algorithm == ScaleAlgorithm.CUBIC:
            return cv2.INTER_CUBIC
        return cv2.INTER_LINEAR


# Global scaler instance
_scaler: Optional[GPUScaler] = None


def get_scaler() -> GPUScaler:
    """Get the global GPU scaler instance."""
    global _scaler
    if _scaler is None:
        _scaler = GPUScaler()
    return _scaler


def scale_image(
    image: np.ndarray,
    scale_factor: float = 2.0,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Convenience function to scale an image."""
    return get_scaler().scale(image, scale_factor, target_size)


if __name__ == "__main__":
    # Test GPU scaling
    import time

    # Create test image
    test_img = np.random.randint(0, 256, (192, 256, 3), dtype=np.uint8)

    scaler = GPUScaler()
    print(f"GPU available: {scaler.gpu_available}")
    print(f"GPU enabled: {scaler.gpu_enabled}")
    print(f"Device: {scaler.device_name}")

    # Test each algorithm
    for algo in ScaleAlgorithm:
        scaler.algorithm = algo

        start = time.time()
        for _ in range(100):
            result = scaler.scale(test_img, 4.0)
        elapsed = time.time() - start

        print(f"{algo.name}: {result.shape}, {elapsed*10:.2f}ms per frame")

    # Compare GPU vs CPU
    if scaler.gpu_available:
        scaler.gpu_enabled = True
        start = time.time()
        for _ in range(100):
            result = scaler.scale(test_img, 4.0)
        gpu_time = time.time() - start

        scaler.gpu_enabled = False
        start = time.time()
        for _ in range(100):
            result = scaler.scale(test_img, 4.0)
        cpu_time = time.time() - start

        print(f"\nGPU: {gpu_time*10:.2f}ms, CPU: {cpu_time*10:.2f}ms")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
