import cv2
import numpy as np
import torch

# Standard 112x112 reference landmarks used in ArcFace & SOTA Face Verification
CANONICAL_5_LANDMARKS_112x112 = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]   # Right mouth corner
], dtype=np.float32)

def align_face_5points(image: np.ndarray, landmarks: np.ndarray, output_size=(112, 112)) -> np.ndarray:
    """
    Applies similarity transformation to align face landmarks with canonical ArcFace 112x112 template.
    image: (H, W, 3) BGR or RGB image
    landmarks: (5, 2) array of detected facial landmarks
    """
    assert landmarks.shape == (5, 2), "Landmarks must have shape (5, 2)"
    ref_pts = CANONICAL_5_LANDMARKS_112x112.copy()
    if output_size != (112, 112):
        scale_x = output_size[0] / 112.0
        scale_y = output_size[1] / 112.0
        ref_pts[:, 0] *= scale_x
        ref_pts[:, 1] *= scale_y

    # Estimate similarity transform matrix (rotation, translation, scale)
    tfm, _ = cv2.estimateAffinePartial2D(landmarks, ref_pts, method=cv2.LMEDS)
    if tfm is None:
        # Fallback if estimation fails
        return cv2.resize(image, output_size)
    aligned = cv2.warpAffine(image, tfm, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return aligned

def preprocess_face_tensor(face_np: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Converts (H, W, 3) image in [0, 255] to PyTorch normalized tensor in [0, 1] of shape (1, 3, H, W).
    """
    if face_np.dtype != np.float32:
        face_np = face_np.astype(np.float32) / 255.0
    tensor = torch.from_numpy(face_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return tensor

def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts (1, 3, H, W) or (3, H, W) PyTorch tensor to (H, W, 3) uint8 image.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img_np = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np

def generate_synthetic_face_samples(num_identities: int = 5, samples_per_identity: int = 4, img_size=(112, 112)):
    """
    Generates realistic structured synthetic facial profiles for reproducible standalone benchmarking.
    Each identity has a unique latent facial signature with structured variations (lighting, slight angle, noise).
    """
    np.random.seed(42)
    identities_data = {}
    
    for id_idx in range(num_identities):
        # Base facial template for identity
        base_color = np.random.uniform(0.5, 0.9, size=(3,))
        feature_seeds = np.random.randn(8)
        samples = []
        
        for s_idx in range(samples_per_identity):
            img = np.zeros((img_size[1], img_size[0], 3), dtype=np.float32)
            # Create facial ellipse
            center = (img_size[0] // 2 + int(feature_seeds[0] * 2), img_size[1] // 2 + int(feature_seeds[1] * 2))
            axes = (int(35 + feature_seeds[2] * 3), int(45 + feature_seeds[3] * 3))
            
            # Draw head
            color = np.clip(base_color + np.random.normal(0, 0.03, 3), 0.1, 1.0)
            cv2.ellipse(img, center, axes, 0, 0, 360, color.tolist(), -1)
            
            # Eyes
            eye_y = int(center[1] - 12 + feature_seeds[4] * 2)
            left_eye_x = int(center[0] - 15)
            right_eye_x = int(center[0] + 15)
            cv2.circle(img, (left_eye_x, eye_y), 4, (0.1, 0.1, 0.1), -1)
            cv2.circle(img, (right_eye_x, eye_y), 4, (0.1, 0.1, 0.1), -1)
            
            # Nose
            nose_y = int(center[1] + 5)
            cv2.line(img, (center[0], nose_y - 5), (center[0], nose_y + 5), (0.2, 0.2, 0.2), 2)
            
            # Mouth
            mouth_y = int(center[1] + 22 + feature_seeds[5] * 2)
            cv2.ellipse(img, (center[0], mouth_y), (10, 4), 0, 0, 180, (0.7, 0.2, 0.2), 2)
            
            # Apply slight intra-personal lighting variation
            lighting_grad = np.linspace(0.8, 1.2, img_size[0]).reshape(1, -1, 1)
            img = np.clip(img * lighting_grad, 0.0, 1.0)
            
            samples.append(img.astype(np.float32))
        identities_data[id_idx] = samples
        
    return identities_data
