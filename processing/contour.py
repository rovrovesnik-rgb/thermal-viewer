"""
MSX-style edge fusion module.
Extracts edges from visible camera and overlays on thermal image.
Implements FLIR MSX-style high-pass filtering (not Canny).
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum


class FusionMode(IntEnum):
    """Fusion modes matching the Android app."""
    THERMAL_ONLY = 0      # No overlay, just thermal
    VISIBLE_ONLY = 1      # Show visible camera only
    EDGE_WHITE = 2        # White edges on thermal (MSX default)
    FUSION_BLEND = 3      # Alpha blend thermal and visible
    PICTURE_IN_PICTURE = 4  # Visible thumbnail in corner
    EDGE_BLACK = 5        # Black edges on thermal


class EdgeMethod(IntEnum):
    """Edge detection method."""
    LAPLACIAN = 0         # Laplacian high-pass (recommended)
    SOBEL = 1             # Sobel gradient magnitude
    HIGH_PASS = 2         # Gaussian high-pass (simplest)


class PipPosition(IntEnum):
    """Picture-in-picture position."""
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3


@dataclass
class FusionSettings:
    """Settings for MSX-style edge fusion."""
    # Enable/disable overlay
    enabled: bool = True

    # Fusion mode (see FusionMode enum)
    mode: FusionMode = FusionMode.EDGE_WHITE

    # Edge detection method
    edge_method: EdgeMethod = EdgeMethod.LAPLACIAN

    # Edge strength multiplier (0.5 - 3.0)
    # Higher = more visible edges
    edge_strength: float = 1.5

    # Edge detail level (kernel size: 1=fine/sharp, 3=medium, 5=coarse/thick)
    # Larger kernel = thicker edges but more noise (auto-compensated)
    edge_detail: int = 1  # Kernel size (1, 3, or 5)

    # Edge threshold (0-50): cuts off low-intensity edges
    # Higher = cleaner edges but may lose fine detail
    edge_threshold: int = 10

    # Fusion alpha for FUSION_BLEND mode (0.0 - 1.0)
    # 0 = thermal only, 1 = visible only
    fusion_alpha: float = 0.3

    # PiP settings
    pip_position: PipPosition = PipPosition.BOTTOM_RIGHT
    pip_size: float = 0.25  # Fraction of thermal image
    pip_opacity: float = 0.9

    # Alignment parameters
    zoom: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    rotation: float = 0.0  # degrees

    # Pre-processing
    # Bilateral filter: preserves edges while smoothing noise
    denoise_strength: int = 9  # Bilateral d parameter (5-15)

    # Edge color for EDGE_WHITE/EDGE_BLACK modes
    # In BGR format
    edge_color: tuple = (255, 255, 255)  # White default

    # Temporal smoothing (1 = off, 2+ = accumulate frames)
    temporal_frames: int = 1


# Keep old name for compatibility
ContourSettings = FusionSettings


class MSXFusion:
    """
    MSX-style thermal/visible fusion processor.

    Uses high-pass filtering (Laplacian/Sobel) instead of Canny edge detection
    for smoother, more natural edge overlay like FLIR MSX.
    """

    THERMAL_WIDTH = 256
    THERMAL_HEIGHT = 192

    def __init__(self, settings: Optional[FusionSettings] = None):
        self.settings = settings or FusionSettings()
        self._edge_history: list[np.ndarray] = []
        self._last_edges: Optional[np.ndarray] = None
        self._last_aligned: Optional[np.ndarray] = None
        self._last_visible_aligned: Optional[np.ndarray] = None

    def extract_edges_msx(self, visible_gray: np.ndarray) -> np.ndarray:
        """
        Extract edges using MSX-style high-pass filtering.

        Unlike Canny (binary edges), this produces gradient magnitudes
        that fade naturally - much better for overlay.

        Args:
            visible_gray: Grayscale image from visible camera

        Returns:
            Edge intensity image (0-255, continuous values)
        """
        # Denoise with bilateral filter (preserves edges)
        d = max(5, min(15, self.settings.denoise_strength))
        denoised = cv2.bilateralFilter(visible_gray, d, 75, 75)

        method = self.settings.edge_method
        strength = max(0.1, self.settings.edge_strength)

        # Get kernel size from settings (1=fine, 3=medium, 5=coarse)
        # ksize=1 uses a special 3x3 approximation that's actually sharper
        ksize = self.settings.edge_detail
        if ksize not in [1, 3, 5]:
            ksize = 3

        # Get edge threshold (cuts off low-intensity edges)
        # Auto-boost threshold for larger kernels to reduce grain
        base_threshold = max(0, min(50, self.settings.edge_threshold))
        threshold = base_threshold + (ksize - 1) * 3  # +0, +6, +12 for sizes 1,3,5

        # Apply extra denoising for larger kernels
        if ksize > 1:
            extra_blur = (ksize - 1) // 2
            if extra_blur > 0:
                denoised = cv2.GaussianBlur(denoised, (0, 0), extra_blur * 0.5)

        if method == EdgeMethod.LAPLACIAN:
            # Laplacian: Non-directional, detects edges in all directions
            # ksize=1 uses 3x3 approximation: [[0,1,0],[1,-4,1],[0,1,0]]
            edges = cv2.Laplacian(denoised, cv2.CV_64F, ksize=ksize)
            edges = np.abs(edges)

        elif method == EdgeMethod.SOBEL:
            # Sobel: Compute gradient magnitude
            # ksize=1 uses simple [-1,0,1] kernel (Scharr-like for ksize=3)
            sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=ksize)
            edges = np.sqrt(sobel_x**2 + sobel_y**2)

        else:  # HIGH_PASS
            # Simplest: Original minus blurred = high frequencies (edges)
            sigma = 1.0 + ksize * 0.5  # Smaller sigma range
            blurred = cv2.GaussianBlur(denoised, (0, 0), sigma)
            edges = cv2.subtract(denoised.astype(np.float64), blurred.astype(np.float64))
            edges = np.abs(edges) * 2  # Boost since this is subtle

        # Apply strength multiplier
        edges = edges * strength

        # Apply threshold - cut off low-intensity edges
        if threshold > 0:
            edges = np.where(edges < threshold, 0, edges - threshold)

        # Normalize to 0-255
        edges = np.clip(edges, 0, 255).astype(np.uint8)

        # Apply temporal smoothing if enabled
        edges = self._apply_temporal_smoothing(edges)

        self._last_edges = edges
        return edges

    def _apply_temporal_smoothing(self, edges: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing using weighted average."""
        temporal_frames = max(1, self.settings.temporal_frames)

        if temporal_frames <= 1:
            self._edge_history.clear()
            return edges

        self._edge_history.append(edges.astype(np.float32))

        while len(self._edge_history) > temporal_frames:
            self._edge_history.pop(0)

        if len(self._edge_history) < 2:
            return edges

        # Weighted average (newer frames have more weight)
        weights = np.arange(1, len(self._edge_history) + 1, dtype=np.float32)
        weights = weights / weights.sum()

        smoothed = np.zeros_like(self._edge_history[0])
        for i, frame in enumerate(self._edge_history):
            smoothed += frame * weights[i]

        return smoothed.astype(np.uint8)

    def align_to_thermal(
        self,
        image: np.ndarray,
        target_width: int = THERMAL_WIDTH,
        target_height: int = THERMAL_HEIGHT
    ) -> np.ndarray:
        """
        Align and scale image to match thermal image dimensions.

        Uses affine transformation for zoom, offset, and rotation.

        Args:
            image: Input image (edges or visible frame)
            target_width: Target width (thermal image width)
            target_height: Target height (thermal image height)

        Returns:
            Aligned image at thermal resolution
        """
        h, w = image.shape[:2]

        zoom = max(0.1, self.settings.zoom)
        offset_x = self.settings.offset_x
        offset_y = self.settings.offset_y
        rotation = self.settings.rotation

        # Source center (with offset applied)
        src_cx = w / 2.0 + offset_x
        src_cy = h / 2.0 + offset_y

        # Destination center
        dst_cx = target_width / 2.0
        dst_cy = target_height / 2.0

        # Scale factors
        scale_x = target_width / w * zoom
        scale_y = target_height / h * zoom

        # Build affine transformation matrix with rotation
        angle_rad = np.radians(rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        m00 = scale_x * cos_a
        m01 = -scale_y * sin_a
        m10 = scale_x * sin_a
        m11 = scale_y * cos_a

        tx = dst_cx - m00 * src_cx - m01 * src_cy
        ty = dst_cy - m10 * src_cx - m11 * src_cy

        M = np.array([[m00, m01, tx], [m10, m11, ty]], dtype=np.float32)

        # Determine interpolation based on image type
        if len(image.shape) == 2:
            # Grayscale
            aligned = cv2.warpAffine(
                image, M, (target_width, target_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            # Color
            aligned = cv2.warpAffine(
                image, M, (target_width, target_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

        return aligned

    def create_edge_overlay(
        self,
        thermal_bgr: np.ndarray,
        edges: np.ndarray,
        white_edges: bool = True
    ) -> np.ndarray:
        """
        Overlay edges on thermal image (MSX style).

        Args:
            thermal_bgr: BGR thermal image
            edges: Edge intensity image (0-255)
            white_edges: True for white edges, False for black

        Returns:
            Composited BGR image
        """
        result = thermal_bgr.copy()

        # Convert edges to 3-channel
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if white_edges:
            # Add edges (brightens where edges are)
            # This is the classic MSX look
            result = cv2.add(result, edges_3ch)
        else:
            # Subtract edges (darkens where edges are)
            result = cv2.subtract(result, edges_3ch)

        return result

    def create_colored_edge_overlay(
        self,
        thermal_bgr: np.ndarray,
        edges: np.ndarray,
        color: tuple = (255, 255, 255)
    ) -> np.ndarray:
        """
        Overlay colored edges on thermal image.

        Args:
            thermal_bgr: BGR thermal image
            edges: Edge intensity image (0-255)
            color: BGR color for edges

        Returns:
            Composited BGR image
        """
        result = thermal_bgr.copy()

        # Normalize edges to 0-1 range for blending
        edge_alpha = edges.astype(np.float32) / 255.0

        # Create colored edge image
        colored_edges = np.zeros_like(thermal_bgr, dtype=np.float32)
        colored_edges[:, :, 0] = color[0] * edge_alpha
        colored_edges[:, :, 1] = color[1] * edge_alpha
        colored_edges[:, :, 2] = color[2] * edge_alpha

        # Blend: result = thermal * (1 - alpha) + colored_edges
        result_float = result.astype(np.float32)
        edge_alpha_3ch = np.stack([edge_alpha] * 3, axis=-1)

        blended = result_float * (1 - edge_alpha_3ch) + colored_edges

        return np.clip(blended, 0, 255).astype(np.uint8)

    def create_fusion_blend(
        self,
        thermal_bgr: np.ndarray,
        visible_bgr: np.ndarray,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Alpha blend thermal and visible images.

        Args:
            thermal_bgr: BGR thermal image
            visible_bgr: BGR visible image (aligned)
            alpha: Blend factor (0=thermal, 1=visible)

        Returns:
            Blended BGR image
        """
        alpha = max(0.0, min(1.0, alpha))
        return cv2.addWeighted(thermal_bgr, 1 - alpha, visible_bgr, alpha, 0)

    def create_pip(
        self,
        thermal_bgr: np.ndarray,
        visible_bgr: np.ndarray,
        position: PipPosition = PipPosition.BOTTOM_RIGHT,
        size: float = 0.25,
        opacity: float = 0.9
    ) -> np.ndarray:
        """
        Create picture-in-picture overlay.

        Args:
            thermal_bgr: BGR thermal image (background)
            visible_bgr: BGR visible image
            position: Corner position
            size: PiP size as fraction of thermal image
            opacity: PiP opacity

        Returns:
            Composited BGR image
        """
        result = thermal_bgr.copy()
        th, tw = thermal_bgr.shape[:2]

        # Calculate PiP dimensions
        pip_w = int(tw * size)
        pip_h = int(th * size)

        # Resize visible to PiP size
        pip_img = cv2.resize(visible_bgr, (pip_w, pip_h), interpolation=cv2.INTER_AREA)

        # Add border
        pip_img = cv2.copyMakeBorder(pip_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        pip_h, pip_w = pip_img.shape[:2]

        # Calculate position
        margin = 5
        if position == PipPosition.TOP_LEFT:
            x, y = margin, margin
        elif position == PipPosition.TOP_RIGHT:
            x, y = tw - pip_w - margin, margin
        elif position == PipPosition.BOTTOM_LEFT:
            x, y = margin, th - pip_h - margin
        else:  # BOTTOM_RIGHT
            x, y = tw - pip_w - margin, th - pip_h - margin

        # Ensure bounds
        x = max(0, min(x, tw - pip_w))
        y = max(0, min(y, th - pip_h))

        # Blend PiP into result
        roi = result[y:y+pip_h, x:x+pip_w]
        blended = cv2.addWeighted(roi, 1 - opacity, pip_img, opacity, 0)
        result[y:y+pip_h, x:x+pip_w] = blended

        return result

    def process_frame(
        self,
        visible_frame: np.ndarray,
        thermal_colored: np.ndarray
    ) -> np.ndarray:
        """
        Complete fusion pipeline based on current mode.

        Args:
            visible_frame: Visible camera frame (grayscale or BGR)
            thermal_colored: BGR thermal image with colormap

        Returns:
            Processed image based on fusion mode
        """
        if not self.settings.enabled:
            return thermal_colored

        mode = self.settings.mode
        th, tw = thermal_colored.shape[:2]

        # Convert visible to grayscale if needed
        if len(visible_frame.shape) == 3:
            visible_gray = cv2.cvtColor(visible_frame, cv2.COLOR_BGR2GRAY)
            visible_bgr = visible_frame
        else:
            visible_gray = visible_frame
            visible_bgr = cv2.cvtColor(visible_frame, cv2.COLOR_GRAY2BGR)

        # Handle each mode
        if mode == FusionMode.THERMAL_ONLY:
            return thermal_colored

        elif mode == FusionMode.VISIBLE_ONLY:
            # Align and return visible camera
            aligned = self.align_to_thermal(visible_bgr, tw, th)
            self._last_visible_aligned = aligned
            return aligned

        elif mode == FusionMode.EDGE_WHITE:
            # MSX-style white edges
            edges = self.extract_edges_msx(visible_gray)
            edges_aligned = self.align_to_thermal(edges, tw, th)
            self._last_aligned = edges_aligned
            return self.create_edge_overlay(thermal_colored, edges_aligned, white_edges=True)

        elif mode == FusionMode.EDGE_BLACK:
            # MSX-style black edges
            edges = self.extract_edges_msx(visible_gray)
            edges_aligned = self.align_to_thermal(edges, tw, th)
            self._last_aligned = edges_aligned
            return self.create_edge_overlay(thermal_colored, edges_aligned, white_edges=False)

        elif mode == FusionMode.FUSION_BLEND:
            # Alpha blend
            visible_aligned = self.align_to_thermal(visible_bgr, tw, th)
            self._last_visible_aligned = visible_aligned
            return self.create_fusion_blend(
                thermal_colored,
                visible_aligned,
                self.settings.fusion_alpha
            )

        elif mode == FusionMode.PICTURE_IN_PICTURE:
            # PiP overlay
            return self.create_pip(
                thermal_colored,
                visible_bgr,
                self.settings.pip_position,
                self.settings.pip_size,
                self.settings.pip_opacity
            )

        # Fallback
        return thermal_colored

    def clear_history(self):
        """Clear temporal smoothing history."""
        self._edge_history.clear()

    @property
    def last_edges(self) -> Optional[np.ndarray]:
        """Get the last extracted edge image."""
        return self._last_edges

    @property
    def last_aligned(self) -> Optional[np.ndarray]:
        """Get the last aligned edge image."""
        return self._last_aligned


# Compatibility aliases
ContourExtractor = MSXFusion
ContourOverlay = MSXFusion


class ContourOverlay:
    """
    High-level interface for contour/fusion overlay functionality.
    Maintains compatibility with existing code.
    """

    def __init__(self):
        self.settings = FusionSettings()
        self._processor = MSXFusion(self.settings)

    def update_settings(self, **kwargs):
        """Update settings with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)

    def process(
        self,
        visible_gray: np.ndarray,
        thermal_colored: np.ndarray
    ) -> np.ndarray:
        """Process visible frame and overlay on thermal image."""
        return self._processor.process_frame(visible_gray, thermal_colored)

    def get_edge_preview(self, visible_gray: np.ndarray) -> np.ndarray:
        """Get just the edge extraction preview."""
        edges = self._processor.extract_edges_msx(visible_gray)
        return self._processor.align_to_thermal(edges)


if __name__ == "__main__":
    # Test MSX fusion
    print("Testing MSX-style fusion...")

    # Create test images
    visible = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(visible, (100, 100), (300, 200), 200, -1)
    cv2.circle(visible, (400, 300), 80, 180, -1)
    cv2.putText(visible, "TEST", (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)

    thermal = np.random.randint(50, 200, (192, 256), dtype=np.uint8)
    thermal_colored = cv2.applyColorMap(thermal, cv2.COLORMAP_JET)

    # Test all modes
    fusion = MSXFusion()

    for mode in FusionMode:
        fusion.settings.mode = mode
        result = fusion.process_frame(visible, thermal_colored)
        filename = f"/tmp/fusion_test_{mode.name.lower()}.png"
        cv2.imwrite(filename, result)
        print(f"Saved {filename}")

    # Test edge methods
    fusion.settings.mode = FusionMode.EDGE_WHITE
    for method in EdgeMethod:
        fusion.settings.edge_method = method
        result = fusion.process_frame(visible, thermal_colored)
        filename = f"/tmp/fusion_test_edge_{method.name.lower()}.png"
        cv2.imwrite(filename, result)
        print(f"Saved {filename}")

    print("Done!")
