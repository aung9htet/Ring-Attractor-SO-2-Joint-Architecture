#!/usr/bin/env python3

import rospy
import csv
import os
from datetime import datetime
from geometry_msgs.msg import WrenchStamped

from tiago_ring_controller.ros.logging import (
    TORQUE_LOG_SCHEMA,
    format_wrench_row,
    log_filename,
    package_root_for_module,
    timestamp_token,
)
from tiago_ring_controller.ros.transport import WRIST_FT_TOPIC

class TorqueRecorder:
    def __init__(self):
        rospy.init_node('torque_recorder', anonymous=True)
        
        # Subscriber to force/torque sensor
        self.ft_sub = rospy.Subscriber(WRIST_FT_TOPIC, WrenchStamped, self.ft_callback)
        
        # Setup logging directory
        package_dir = package_root_for_module(__file__)
        self.output_dir = os.path.join(
            package_dir,
            'experiment_results',
            TORQUE_LOG_SCHEMA.directory_name,
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp = timestamp_token(datetime.now())
        self.log_path = os.path.join(
            self.output_dir,
            log_filename(TORQUE_LOG_SCHEMA, timestamp),
        )
        
        self.log_file = open(self.log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(TORQUE_LOG_SCHEMA.columns)
        self.log_file.flush()
        
        self.start_time = rospy.get_time()
        self.sample_count = 0
        
        rospy.loginfo("Torque Recorder initialized")
        rospy.loginfo(f"Recording torque data to: {self.log_path}")
    
    def ft_callback(self, msg):
        """Callback for force/torque sensor data"""
        current_time = rospy.get_time()
        elapsed_time = current_time - self.start_time
        
        # Log sensor data
        self.csv_writer.writerow(
            format_wrench_row(
                elapsed_time,
                (
                    msg.wrench.force.x,
                    msg.wrench.force.y,
                    msg.wrench.force.z,
                ),
                (
                    msg.wrench.torque.x,
                    msg.wrench.torque.y,
                    msg.wrench.torque.z,
                ),
            )
        )
        self.log_file.flush()
        self.sample_count += 1
    
    def shutdown(self):
        """Cleanup on shutdown"""
        self.log_file.close()
        rospy.loginfo(f"Torque recording stopped ({self.sample_count} samples recorded)")
        
        if os.path.exists(self.log_path):
            file_size = os.path.getsize(self.log_path) / 1024
            rospy.loginfo(f"Torque data saved to: {self.log_path} ({file_size:.2f} KB)")
        else:
            rospy.logerr(f"Log file not found at: {self.log_path}")

def main():
    try:
        recorder = TorqueRecorder()
        rospy.on_shutdown(recorder.shutdown)
        rospy.spin()
    except Exception as e:
        rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    main()
