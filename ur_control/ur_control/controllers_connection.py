#!/usr/bin/env python

import time

import rclpy
from controller_manager_msgs.srv import (SwitchController, LoadController,
                                          UnloadController, ListControllers)
from ur_control.utils import solve_namespace, read_parameter


# ros2_control lifecycle state name for an activated controller (ROS 1 used "running").
ACTIVE = "active"


class ControllersConnection():
    """Wrapper around the ros2_control controller_manager services.

    Requires a spinning rclpy node (e.g. driven by a MultiThreadedExecutor in a
    background thread): the synchronous ``client.call`` calls below rely on the
    executor processing the service responses.
    """

    def __init__(self, node, namespace=None):
        self.node = node
        self.controllers_list = []

        self.ns = solve_namespace(namespace, node=node)[1:]

        if namespace:
            prefix = namespace + 'controller_manager/'
        else:
            prefix = '/controller_manager/'

        # Only for osx sim: the controller_manager lives at the global namespace.
        self.use_sim = bool(read_parameter(node, "use_gazebo_sim", False))
        if self.use_sim:
            prefix = '/controller_manager/'

        self.switch_client = node.create_client(SwitchController, prefix + 'switch_controller')
        self.load_client = node.create_client(LoadController, prefix + 'load_controller')
        self.unload_client = node.create_client(UnloadController, prefix + 'unload_controller')
        self.list_controllers_client = node.create_client(ListControllers, prefix + 'list_controllers')
        self.list_controllers_client.wait_for_service(timeout_sec=1.0)

    def _call(self, client, request, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("Service %s unavailable" % client.srv_name)
        return client.call(request)

    def get_loaded_controllers(self):
        self.controllers_list = []
        try:
            result = self._call(self.list_controllers_client, ListControllers.Request())
            for controller in result.controller:
                self.controllers_list.append(controller.name)
        except Exception:
            pass

    def get_controller_state(self, controller_name):
        try:
            result = self._call(self.list_controllers_client, ListControllers.Request())
            for controller in result.controller:
                if controller.name == controller_name:
                    return controller.state
        except Exception:
            pass
        self.node.get_logger().error("Controller %s not found" % controller_name)
        raise ValueError(f"Controller {controller_name} not found")

    def load_controllers(self, controllers_list):
        self.get_loaded_controllers()
        for controller in controllers_list:
            if controller not in self.controllers_list:
                result = self._call(self.load_client, LoadController.Request(name=controller), timeout=0.1)
                self.node.get_logger().info('Loading controller %s. Result=%s' % (controller, result.ok))
                if result.ok:
                    self.controllers_list.append(controller)

    def unload_controllers(self, controllers_list):
        try:
            for controller in controllers_list:
                result = self._call(self.unload_client, UnloadController.Request(name=controller), timeout=0.1)
                self.node.get_logger().info('Unloading controller %s. Result=%s' % (controller, result.ok))
                if result.ok:
                    self.controllers_list.remove(controller)
        except Exception as e:
            self.node.get_logger().error("Unload controllers service call failed: %s" % e)

    def check_on_controllers_state(self, on_controllers):
        """
        Return a list of controllers that need to be activated (not already active)
        """
        controllers_to_switch = []
        for c in on_controllers:
            if c is not None and self.get_controller_state(c) != ACTIVE:
                controllers_to_switch.append(c)
        return controllers_to_switch

    def check_off_controllers_state(self, off_controllers):
        """
        Return a list of controllers that need to be deactivated (currently active)
        """
        controllers_to_switch = []
        for c in off_controllers:
            if c is not None:
                state = self.get_controller_state(c)
                if state == ACTIVE:
                    controllers_to_switch.append(c)
        return controllers_to_switch

    def switch_controllers(self, controllers_on, controllers_off,
                           strictness=1):
        """
        Give the controllers you want to switch on or off.
        :param controllers_on: ["name_controller_1", "name_controller2",...,"name_controller_n"]
        :param controllers_off: ["name_controller_1", "name_controller2",...,"name_controller_n"]
        :return:
        """

        if self.use_sim:
            controllers_on = [self.ns + controller for controller in controllers_on]
            controllers_off = [self.ns + controller for controller in controllers_off]

        try:
            request = SwitchController.Request()
            request.activate_controllers = self.check_on_controllers_state(controllers_on)
            request.deactivate_controllers = self.check_off_controllers_state(controllers_off)
            request.strictness = strictness

            if request.activate_controllers == [] and request.deactivate_controllers == []:
                return True

            switch_result = self._call(self.switch_client, request, timeout=0.1)
            """
            [controller_manager_msgs/srv/SwitchController]
            int32 BEST_EFFORT=1
            int32 STRICT=2
            string[] activate_controllers
            string[] deactivate_controllers
            int32 strictness
            bool activate_asap
            builtin_interfaces/Duration timeout
            ---
            bool ok
            """
            self.node.get_logger().debug("Switch Result==>" + str(switch_result.ok))

            if not switch_result.ok:  # Return if service failed
                return False

            # Check that the controllers are active before returning
            start_time = time.time()
            while (time.time() - start_time) < 5.0 and rclpy.ok():
                all_active = True
                for controller in controllers_on:
                    if self.get_controller_state(controller) != ACTIVE:
                        all_active = False
                        break
                if all_active:
                    break
            time.sleep(1.0)  # wait for the controller to be fully activated
            return switch_result.ok

        except Exception as e:
            self.node.get_logger().error("Switch controllers service call failed: %s" % e)
            return False

    def reset_controllers(self):
        """
        We turn on and off the given controllers
        :param controllers_reset: ["name_controller_1", "name_controller2",...,"name_controller_n"]
        :return:
        """

        reset_result = False

        result_off_ok = self.switch_controllers(controllers_on=[], controllers_off=self.controllers_list)

        self.node.get_logger().debug("Deactivated Controllers")

        if result_off_ok:
            self.node.get_logger().debug("Activating Controllers")
            result_on_ok = self.switch_controllers(
                controllers_on=self.controllers_list, controllers_off=[])
            if result_on_ok:
                self.node.get_logger().debug("Controllers Reset==>" +
                                             str(self.controllers_list))
                reset_result = True
            else:
                self.node.get_logger().debug("result_on_ok==>" + str(result_on_ok))
        else:
            self.node.get_logger().debug("result_off_ok==>" + str(result_off_ok))

        return reset_result

    def update_controllers_list(self, new_controllers_list):

        self.controllers_list = new_controllers_list
