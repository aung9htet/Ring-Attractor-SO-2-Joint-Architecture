#!/usr/bin/env python3

import argparse
import time

import rospy
from sensor_msgs.msg import Image

from tiago_ring_controller.ros.camera import (
	CAMERA_BUFFER_SIZE,
	CAMERA_LOOP_HZ,
	CAMERA_QUEUE_SIZE,
	DEFAULT_RECORDING_CAMERA_TOPIC,
	resolve_image_topic,
)


class CameraStream:
	"""Subscribe to a ROS image topic and display or monitor frames."""

	def __init__(self, topic: str = None, window_name: str = "TIAGo Camera", headless: bool = False):
		self._cv2 = None
		self._CvBridge = None
		self.bridge = None

		self.headless = bool(headless)
		self.window_name = window_name
		self.topic = topic

		self.latest_frame = None
		self.frame_count = 0
		self.start_time = None
		self.last_log_time = 0.0
		self.last_stamp = None

		if not self.headless:
			try:
				import cv2  # pylint: disable=import-outside-toplevel
				from cv_bridge import CvBridge  # pylint: disable=import-outside-toplevel

				self._cv2 = cv2
				self._CvBridge = CvBridge
				self.bridge = CvBridge()
			except Exception as exc:  # pragma: no cover
				raise RuntimeError(
					"Display mode requires OpenCV and cv_bridge. "
					"Use --headless if display is unavailable."
				) from exc
		else:
			try:
				from cv_bridge import CvBridge  # pylint: disable=import-outside-toplevel

				self._CvBridge = CvBridge
				self.bridge = CvBridge()
			except Exception:
				self.bridge = None

	def _resolve_topic(self) -> str:
		return resolve_image_topic(self.topic, rospy.get_published_topics())

	def _image_callback(self, msg: Image):
		self.frame_count += 1
		self.last_stamp = msg.header.stamp.to_sec() if msg.header else None

		if self.bridge is not None:
			try:
				self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
			except Exception:
				self.latest_frame = None

	def run(self, duration: float = 0.0, fps_log_interval: float = 1.0):
		topic = self._resolve_topic()
		rospy.loginfo("Streaming camera topic: %s", topic)

		sub = rospy.Subscriber(
			topic,
			Image,
			self._image_callback,
			queue_size=CAMERA_QUEUE_SIZE,
			buff_size=CAMERA_BUFFER_SIZE,
		)

		self.start_time = time.time()
		self.last_log_time = self.start_time
		rate = rospy.Rate(CAMERA_LOOP_HZ)

		try:
			while not rospy.is_shutdown():
				now = time.time()

				if duration > 0.0 and (now - self.start_time) >= duration:
					rospy.loginfo("Duration reached (%.1fs). Exiting.", duration)
					break

				if (now - self.last_log_time) >= fps_log_interval:
					elapsed = max(now - self.start_time, 1e-6)
					fps = self.frame_count / elapsed
					rospy.loginfo("Frames: %d | Avg FPS: %.2f", self.frame_count, fps)
					self.last_log_time = now

				if not self.headless and self.latest_frame is not None:
					self._cv2.imshow(self.window_name, self.latest_frame)
					key = self._cv2.waitKey(1) & 0xFF
					if key in (27, ord("q")):
						rospy.loginfo("User requested exit.")
						break

				rate.sleep()
		finally:
			sub.unregister()
			if self._cv2 is not None:
				self._cv2.destroyAllWindows()


def _parse_args():
	parser = argparse.ArgumentParser(description="Stream TIAGo camera topic from ROS/Gazebo.")
	parser.add_argument(
		"--topic",
		default="",
		help=(
			"Image topic. Auto-detected if omitted. "
			"Auto mode prefers /recording_camera/image_raw."
		),
	)
	parser.add_argument("--headless", action="store_true", help="Run without display window.")
	parser.add_argument("--duration", type=float, default=0.0, help="Stop automatically after this many seconds.")
	parser.add_argument(
		"--fps-log-interval",
		type=float,
		default=1.0,
		help="Seconds between FPS logs.",
	)
	parser.add_argument("--window-name", default="TIAGo Camera", help="OpenCV window title.")
	return parser.parse_args()


def main():
	args = _parse_args()
	rospy.init_node("experiment_camera_stream", anonymous=True)

	streamer = CameraStream(
		topic=args.topic or None,
		window_name=args.window_name,
		headless=args.headless,
	)
	streamer.run(duration=args.duration, fps_log_interval=args.fps_log_interval)


if __name__ == "__main__":
	main()
