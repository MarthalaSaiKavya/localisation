#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
import numpy as np, math

class OdometryMotionModel:
    def __init__(self, alphas=None):
        self.alphas = alphas or [0.02, 0.02, 0.02, 0.02]
        self.prev = None
        self.pub = rospy.Publisher("~delta_pose", Pose2D, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.cb_odom)

    @staticmethod
    def yaw_from_quat(q):
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def sample(self, mean, var):
        return np.random.normal(mean, math.sqrt(max(var, 1e-12)))

    def cb_odom(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q  = msg.pose.pose.orientation
        th = self.yaw_from_quat(q)
        curr = np.array([px, py, th], dtype=float)
        if self.prev is None:
            self.prev = curr; return

        x0, y0, th0 = self.prev
        x1, y1, th1 = curr
        dx = x1 - x0; dy = y1 - y0
        trans = math.hypot(dx, dy)
        rot1  = math.atan2(dy, dx) - th0; rot1 = math.atan2(math.sin(rot1), math.cos(rot1))
        rot2  = th1 - th0 - rot1;         rot2 = math.atan2(math.sin(rot2), math.cos(rot2))

        a1, a2, a3, a4 = self.alphas
        var_rot1  = a1*rot1*rot1 + a2*trans*trans
        var_trans = a3*trans*trans + a4*(rot1*rot1 + rot2*rot2)
        var_rot2  = a1*rot2*rot2 + a2*trans*trans

        rot1_hat  = self.sample(rot1,  var_rot1)
        trans_hat = self.sample(trans, var_trans)
        rot2_hat  = self.sample(rot2,  var_rot2)

        dx_hat = trans_hat * math.cos(th0 + rot1_hat)
        dy_hat = trans_hat * math.sin(th0 + rot1_hat)
        dth_hat = rot2_hat
        self.pub.publish(Pose2D(x=dx_hat, y=dy_hat, theta=dth_hat))
        self.prev = curr

def main():
    rospy.init_node("motion_model_node")
    OdometryMotionModel()
    rospy.loginfo("Motion model running; listening to /odom")
    rospy.spin()

if __name__ == "__main__":
    main()
