import cv2
import torch
import mediapipe as mp


def test_imports():
    """Check core libraries are installed and importable."""
    assert cv2.__version__ is not None
    assert torch.__version__ is not None
    assert mp.__version__ is not None


def test_dummy_frame_to_tensor():
    """Check that a dummy frame can be created and converted to a tensor."""
    # Create a dummy frame (black image)
    frame = cv2.imread("tests/assets/black.jpg")
    if frame is None:
        # fallback if no image exists → just make a black array
        frame = (255 * torch.zeros((480, 640, 3), dtype=torch.uint8).numpy())

    # Resize for model input
    resized = cv2.resize(frame, (224, 224))

    # Convert to tensor
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

    # Check shape is (3, 224, 224)
    assert tensor.shape == (3, 224, 224)


def test_mediapipe_hands():
    """Check that Mediapipe Hands can be initialized."""
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    assert mp_hands is not None
    mp_hands.close()
