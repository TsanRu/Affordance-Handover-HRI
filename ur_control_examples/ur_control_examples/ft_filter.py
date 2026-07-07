#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2018-2023 Cristian Beltran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Cristian Beltran

import argparse
import collections
import sys
import time

import numpy as np
import rclpy
from rclpy.parameter import Parameter
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from std_srvs.srv import Empty, SetBool

from ur_control import conversions, filters, spalg, utils


class FTFilterNode(Node):

    def __init__(self, in_topic, namespace="", out_topic=None,
                 sampling_frequency=500, cutoff=10,
                 order=2, data_window=10,
                 republish=False):
        super().__init__('ft_filter')

        self.enable_publish = republish
        self.enable_filtering = True

        ns = utils.solve_namespace(namespace, node=self) if namespace else self.get_namespace()
        if not ns.endswith('/'):
            ns += '/'

        in_topic = in_topic.lstrip('/')
        self.in_topic = ns + in_topic

        if out_topic:
            out_topic = out_topic.lstrip('/')
            self.out_topic = ns + out_topic
        else:
            self.out_topic = self.in_topic.rstrip('/') + '/filtered'

        self.out_tcp_topic = self.out_topic.rstrip('/') + 'tcp'
        self._offset_param = self.out_topic.replace('/', '_').strip('_') + '_ft_offset'

        self.get_logger().info('Publishing filtered FT to %s' % self.out_topic)

        offset = self.declare_parameter(self._offset_param, [0.0] * 6).value
        self.wrench_offset = np.zeros(6) if offset is None else np.array(offset, dtype=float)

        self.pub = self.create_publisher(WrenchStamped, self.out_topic, 1)
        self.pub_tcp = self.create_publisher(WrenchStamped, self.out_tcp_topic, 1)

        self.create_service(Empty, self.out_topic.rstrip('/') + '/zero_ftsensor', self._srv_zeroing)
        self.create_service(SetBool, self.out_topic.rstrip('/') + '/enable_publish', self._srv_publish)
        self.create_service(SetBool, self.out_topic.rstrip('/') + '/enable_filtering', self._srv_filtering)

        self.filter = filters.ButterLowPass(cutoff, sampling_frequency, order)

        self.data_window = data_window
        assert self.data_window >= 5
        self.data_queue = collections.deque(maxlen=self.data_window)

        self.create_subscription(
            WrenchStamped, self.in_topic, self.raw_wrench_cb, qos_profile_sensor_data)

        self.get_logger().info('FT filter successfully initialized')

    def add_wrench_observation(self, wrench):
        self.data_queue.append(np.array(wrench))

    def raw_wrench_cb(self, msg):
        if not rclpy.ok():
            return
        current_wrench = conversions.from_wrench(msg.wrench)
        self.add_wrench_observation(current_wrench)
        if not self.enable_publish:
            return

        if self.enable_filtering:
            current_wrench = self.get_filtered_wrench()

        if current_wrench is None:
            return

        data = current_wrench - self.wrench_offset
        out_msg = WrenchStamped()
        out_msg.wrench = conversions.to_wrench(data)
        self.pub.publish(out_msg)

        tcp_param = self.out_tcp_topic.replace('/', '_').strip('_') + '_pose_sensor_to_tcp'
        if self.has_parameter(tcp_param):
            pose_sensor_to_tcp = self.get_parameter(tcp_param).value
            if pose_sensor_to_tcp is not None:
                tcp_wrench = data.copy()
                tcp_wrench[:3] += spalg.sensor_torque_to_tcp_force(
                    tcp_position=pose_sensor_to_tcp, sensor_torques=current_wrench[3:])
                tcp_wrench[3:] = np.zeros(3)
                tcp_msg = WrenchStamped()
                tcp_msg.wrench = conversions.to_wrench(tcp_wrench)
                self.pub_tcp.publish(tcp_msg)

    def get_filtered_wrench(self):
        if len(self.data_queue) < self.data_window:
            return None
        wrench_filtered = self.filter(np.array(self.data_queue))
        return wrench_filtered[-1, :]

    def update_wrench_offset(self):
        current_wrench = self.get_filtered_wrench()
        if current_wrench is not None:
            self.wrench_offset = current_wrench
            self.set_parameters([Parameter(
                self._offset_param, Parameter.Type.DOUBLE_ARRAY,
                self.wrench_offset.tolist())])

    def set_enable_publish(self, enable):
        self.enable_publish = enable

    def set_enable_filtering(self, enable):
        self.enable_filtering = enable

    def _srv_zeroing(self, _request, response):
        self.update_wrench_offset()
        return response

    def _srv_publish(self, request, response):
        self.set_enable_publish(request.data)
        response.success = True
        return response

    def _srv_filtering(self, request, response):
        self.set_enable_filtering(request.data)
        response.success = True
        return response


def main(args=None):
    """Filter and republish FT sensor data."""
    parser = argparse.ArgumentParser(description='Filter FT signal')
    parser.add_argument('-ns', '--namespace', type=str, default='')
    parser.add_argument('-t', '--ft_topic', type=str, required=True,
                        help='FT sensor data topic (relative to namespace)')
    parser.add_argument('-ot', '--out_topic', type=str,
                        help='Topic where filtered data will be published')
    parser.add_argument('-z', '--zero', action='store_true', help='Zero FT signal')

    argv = remove_ros_args(args if args is not None else sys.argv)
    cli_args = parser.parse_args(argv[1:])

    rclpy.init(args=args)
    node = FTFilterNode(
        namespace=cli_args.namespace,
        in_topic=cli_args.ft_topic,
        out_topic=cli_args.out_topic,
        republish=True)

    time.sleep(1.0)
    if cli_args.zero:
        node.update_wrench_offset()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
