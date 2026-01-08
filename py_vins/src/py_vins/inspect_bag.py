#!/usr/bin/env python3
import rosbag
import rospy
from std_msgs.msg import Header

# This script only inspects the message structure
bag_path = "/root/datasets/miluv/1a/ifo001_uncompressed.bag"
topic_name = "/ifo001/uwb/range"

# Open the bag
with rosbag.Bag(bag_path, "r") as bag:
    print(f"Reading messages from {topic_name} in {bag_path}...\n")
    
    count = 0
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        count += 1
        
        # Print top-level fields
        print(f"Message {count} at t={t.to_sec():.6f}:")
        try:
            # Access fields dynamically
            for field in msg.__slots__:
                value = getattr(msg, field)
                print(f"  {field}: {value}")
        except Exception as e:
            print(f"  Could not read message fields: {e}")
        
        if count >= 5:  # limit output for quick inspection
            break

    print(f"\nTotal messages read: {count}")