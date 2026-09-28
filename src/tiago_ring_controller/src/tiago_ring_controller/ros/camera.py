"""ROS-independent camera topic selection contracts."""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


DEFAULT_RECORDING_CAMERA_TOPIC = "/recording_camera/image_raw"
IMAGE_MESSAGE_TYPE = "sensor_msgs/Image"
CAMERA_QUEUE_SIZE = 1
CAMERA_BUFFER_SIZE = 2 ** 24
CAMERA_LOOP_HZ = 60

PREFERRED_IMAGE_TOPICS: Tuple[str, ...] = (
    DEFAULT_RECORDING_CAMERA_TOPIC,
    "/xtion/rgb/image_raw",
    "/xtion/rgb/image_color",
    "/camera/rgb/image_raw",
    "/camera/image_raw",
)


@dataclass(frozen=True)
class CameraSubscriptionContract:
    message_type: str = IMAGE_MESSAGE_TYPE
    queue_size: int = CAMERA_QUEUE_SIZE
    buffer_size: int = CAMERA_BUFFER_SIZE
    loop_hz: int = CAMERA_LOOP_HZ


CAMERA_SUBSCRIPTION_CONTRACT = CameraSubscriptionContract()


def resolve_image_topic(
    explicit_topic: Optional[str],
    published_topics: Iterable[Sequence[str]],
) -> str:
    """Mirror ``CameraStream._resolve_topic`` exactly.

    An explicitly supplied non-empty topic is returned without checking the
    advertised topic list.  Automatic mode keeps only exact
    ``sensor_msgs/Image`` types, follows the fixed preference order, then
    chooses the first lexicographically sorted lowercase ``rgb``/``color``
    candidate, and finally the first sorted image topic.
    """

    if explicit_topic:
        return explicit_topic

    image_topics = [
        name
        for name, message_type in published_topics
        if message_type == IMAGE_MESSAGE_TYPE
    ]
    if not image_topics:
        raise RuntimeError("No sensor_msgs/Image topics found.")

    for topic in PREFERRED_IMAGE_TOPICS:
        if topic in image_topics:
            return topic

    rgb_candidates = [
        topic for topic in image_topics if "rgb" in topic or "color" in topic
    ]
    if rgb_candidates:
        return sorted(rgb_candidates)[0]
    return sorted(image_topics)[0]


__all__ = [
    "CAMERA_BUFFER_SIZE",
    "CAMERA_LOOP_HZ",
    "CAMERA_QUEUE_SIZE",
    "CAMERA_SUBSCRIPTION_CONTRACT",
    "CameraSubscriptionContract",
    "DEFAULT_RECORDING_CAMERA_TOPIC",
    "IMAGE_MESSAGE_TYPE",
    "PREFERRED_IMAGE_TOPICS",
    "resolve_image_topic",
]
