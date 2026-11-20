"""ROS interface doubles let adapter tests run without a ROS installation."""
import sys
import types
import os
from pathlib import Path


if os.environ.get("AIVA_TEST_WITH_ROS") != "1" and "rclpy" not in sys.modules:
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *args, **kwargs: None
    rclpy.spin = lambda *args, **kwargs: None
    rclpy.shutdown = lambda *args, **kwargs: None
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = type("Node", (), {})
    rclpy.node = rclpy_node
    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = rclpy_node

if os.environ.get("AIVA_TEST_WITH_ROS") != "1" and "cryptic_signal_core.msg" not in sys.modules:
    cryptic = types.ModuleType("cryptic_signal_core")
    cryptic.__path__ = [str(Path(__file__).resolve().parents[2] / 'cypher/cryptic_signal_core')]
    cryptic_msg = types.ModuleType("cryptic_signal_core.msg")
    cryptic_msg.TelepathicSignal = type("TelepathicSignal", (), {})
    cryptic.msg = cryptic_msg
    sys.modules["cryptic_signal_core"] = cryptic
    sys.modules["cryptic_signal_core.msg"] = cryptic_msg

if os.environ.get("AIVA_TEST_WITH_ROS") != "1" and "std_msgs.msg" not in sys.modules:
    std_msgs = types.ModuleType("std_msgs")
    std_msgs.__path__ = []
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = type("String", (), {"__init__": lambda self: setattr(self, "data", "")})
    std_msgs.msg = std_msgs_msg
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs_msg

if os.environ.get("AIVA_TEST_WITH_ROS") != "1" and "ament_index_python.packages" not in sys.modules:
    ament_index = types.ModuleType("ament_index_python")
    ament_index.__path__ = []
    packages = types.ModuleType("ament_index_python.packages")
    packages.get_package_share_directory = lambda _name: (_ for _ in ()).throw(LookupError())
    ament_index.packages = packages
    sys.modules["ament_index_python"] = ament_index
    sys.modules["ament_index_python.packages"] = packages
