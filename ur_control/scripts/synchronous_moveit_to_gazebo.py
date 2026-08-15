import rospy
from moveit_commander import RobotCommander
import moveit_commander

rospy.init_node("update_moveit_state")
robot = RobotCommander()
robot.get_current_state()  # 讓 MoveIt! 更新機械臂狀態

group_left = moveit_commander.MoveGroupCommander("rightarm")  # 請確認是你的 MoveIt! 群組
group_right = moveit_commander.MoveGroupCommander("leftarm")  # 請確認是你的 MoveIt! 群組
group_left.set_named_target("home")  # MoveIt! 預設的 home 位置
group_left.go()
group_right.set_named_target("home")  # MoveIt! 預設的 home 位置
group_right.go()