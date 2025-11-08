#!/usr/bin/env python3
import math
import os
import threading

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from pf_localization.map_utils import load_likelihood_field


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class ParticleFilterNode:
    def __init__(self):
        pkg_dir = rospy.get_param(
            "~pkg_dir", os.path.expanduser("~/catkin_ws/src/pf_localization")
        )
        map_yaml = rospy.get_param(
            "~map_yaml", os.path.join(pkg_dir, "maps", "house_map.yaml")
        )
        self.num_particles = int(rospy.get_param("~num_particles", 500))
        self.alpha1 = float(rospy.get_param("~alpha1", 0.02))
        self.alpha2 = float(rospy.get_param("~alpha2", 0.02))
        self.alpha3 = float(rospy.get_param("~alpha3", 0.02))
        self.alpha4 = float(rospy.get_param("~alpha4", 0.02))
        self.sigma_hit = float(rospy.get_param("~sigma_hit", 0.2))
        self.sigma_hit_sq = max(self.sigma_hit ** 2, 1e-6)
        self.z_hit = float(rospy.get_param("~z_hit", 0.8))
        self.z_rand = float(rospy.get_param("~z_rand", 0.2))
        self.beam_step = int(rospy.get_param("~beam_step", 8))
        self.resample_noise_xy = float(rospy.get_param("~resample_noise_xy", 0.01))
        self.resample_noise_theta = float(
            rospy.get_param("~resample_noise_theta", 0.01)
        )
        self.publish_tf = bool(rospy.get_param("~publish_tf", True))
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")

        self.rng = np.random.default_rng()
        self.field = load_likelihood_field(map_yaml)
        self.particles = self.field.sample_free_states(self.num_particles, self.rng)
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.prev_odom = None
        self.lock = threading.Lock()

        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=1)
        self.odom_pub = rospy.Publisher("~odom", Odometry, queue_size=1)
        self.cloud_pub = rospy.Publisher("~particle_cloud", PoseArray, queue_size=1)
        self.tf_br = tf.TransformBroadcaster() if self.publish_tf else None

        rospy.Subscriber("/odom", Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.scan_cb, queue_size=1)

        rospy.loginfo(
            "Particle filter ready with %d particles (%s)",
            self.num_particles,
            map_yaml,
        )

    @staticmethod
    def yaw_from_quaternion(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_cb(self, msg):
        pose = msg.pose.pose
        curr = np.array(
            [
                pose.position.x,
                pose.position.y,
                self.yaw_from_quaternion(pose.orientation),
            ],
            dtype=float,
        )

        with self.lock:
            if self.prev_odom is None:
                self.prev_odom = curr
                return
            delta = self.compute_odometry_delta(self.prev_odom, curr)
            self.prev_odom = curr
            if delta is None:
                return
            self.apply_motion_model(delta)

    def compute_odometry_delta(self, prev, curr):
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        trans = math.hypot(dx, dy)
        if trans < 1e-6 and abs(wrap_angle(curr[2] - prev[2])) < 1e-6:
            return None
        rot1 = wrap_angle(math.atan2(dy, dx) - prev[2]) if trans >= 1e-6 else 0.0
        rot2 = wrap_angle(curr[2] - prev[2] - rot1)
        return rot1, trans, rot2

    def apply_motion_model(self, delta):
        rot1, trans, rot2 = delta
        n = self.num_particles
        var_rot1 = abs(self.alpha1 * rot1 * rot1 + self.alpha2 * trans * trans)
        var_trans = abs(self.alpha3 * trans * trans + self.alpha4 * (rot1 * rot1 + rot2 * rot2))
        var_rot2 = abs(self.alpha1 * rot2 * rot2 + self.alpha2 * trans * trans)

        rot1_hat = self.rng.normal(rot1, math.sqrt(max(var_rot1, 1e-9)), size=n)
        trans_hat = self.rng.normal(trans, math.sqrt(max(var_trans, 1e-9)), size=n)
        rot2_hat = self.rng.normal(rot2, math.sqrt(max(var_rot2, 1e-9)), size=n)

        headings = self.particles[:, 2] + rot1_hat
        self.particles[:, 0] += trans_hat * np.cos(headings)
        self.particles[:, 1] += trans_hat * np.sin(headings)
        self.particles[:, 2] = wrap_angle(self.particles[:, 2] + rot1_hat + rot2_hat)

        # Keep particles inside known free space when possible.
        occupied = np.array(
            [self.field.is_occupied(x, y) for x, y in self.particles[:, :2]], dtype=bool
        )
        if np.any(occupied):
            replacements = self.field.sample_free_states(int(np.sum(occupied)), self.rng)
            self.particles[occupied] = replacements

    def scan_cb(self, scan):
        with self.lock:
            if self.prev_odom is None:
                return
            weights = self.compute_sensor_weights(scan)
            if weights is None:
                return
            self.weights = weights
            self.resample()
            stamp = scan.header.stamp
            if stamp == rospy.Time():
                stamp = rospy.Time.now()
            self.publish_outputs(stamp)

    def compute_sensor_weights(self, scan):
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        total_beams = len(ranges)
        step = max(1, self.beam_step)
        indices = np.arange(0, total_beams, step, dtype=np.int32)
        if indices.size == 0:
            return None

        angles = scan.angle_min + indices * scan.angle_increment
        weights_log = np.zeros(self.num_particles, dtype=np.float64)
        valid_beams = 0
        rand_term = self.z_rand * (1.0 / max(scan.range_max, 1e-3))

        for idx, angle in zip(indices, angles):
            r = ranges[idx]
            if not np.isfinite(r):
                continue
            if r < scan.range_min or r > scan.range_max:
                continue
            global_angles = self.particles[:, 2] + angle
            hit_x = self.particles[:, 0] + r * np.cos(global_angles)
            hit_y = self.particles[:, 1] + r * np.sin(global_angles)
            dists = self.field.lookup_distances(hit_x, hit_y)
            prob = self.z_hit * np.exp(-0.5 * (dists ** 2) / self.sigma_hit_sq) + rand_term
            np.clip(prob, 1e-9, None, out=prob)
            weights_log += np.log(prob)
            valid_beams += 1

        if valid_beams == 0:
            return None

        # Normalize weights in log space for numerical stability.
        weights = np.exp(weights_log - np.max(weights_log))
        total = np.sum(weights)
        if not np.isfinite(total) or total <= 0.0:
            rospy.logwarn("Sensor update produced invalid weights; skipping.")
            return None
        return weights / total

    def resample(self):
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0  # guard against float precision
        step = 1.0 / self.num_particles
        r = self.rng.random() * step
        indexes = np.zeros(self.num_particles, dtype=np.int32)
        i = 0
        for m in range(self.num_particles):
            u = r + m * step
            while u > cumulative[i]:
                i += 1
            indexes[m] = i
        self.particles = self.particles[indexes]
        self.weights = np.ones(self.num_particles) / self.num_particles

        if self.resample_noise_xy > 0 or self.resample_noise_theta > 0:
            noise_xy = self.rng.normal(
                0.0, self.resample_noise_xy, size=(self.num_particles, 2)
            )
            noise_theta = self.rng.normal(
                0.0, self.resample_noise_theta, size=self.num_particles
            )
            self.particles[:, 0] += noise_xy[:, 0]
            self.particles[:, 1] += noise_xy[:, 1]
            self.particles[:, 2] = wrap_angle(self.particles[:, 2] + noise_theta)

        occupied = np.array(
            [self.field.is_occupied(x, y) for x, y in self.particles[:, :2]], dtype=bool
        )
        if np.any(occupied):
            replacements = self.field.sample_free_states(int(np.sum(occupied)), self.rng)
            self.particles[occupied] = replacements

    def estimate_pose(self):
        weights = self.weights
        mean_x = np.average(self.particles[:, 0], weights=weights)
        mean_y = np.average(self.particles[:, 1], weights=weights)
        mean_cos = np.average(np.cos(self.particles[:, 2]), weights=weights)
        mean_sin = np.average(np.sin(self.particles[:, 2]), weights=weights)
        mean_theta = math.atan2(mean_sin, mean_cos)
        return mean_x, mean_y, mean_theta

    def publish_outputs(self, stamp):
        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = self.map_frame
        pose_array.poses = [self._particle_to_pose(p) for p in self.particles]
        self.cloud_pub.publish(pose_array)

        mean_x, mean_y, mean_theta = self.estimate_pose()
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.map_frame
        pose_msg.pose = self._particle_to_pose((mean_x, mean_y, mean_theta))
        self.pose_pub.publish(pose_msg)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.map_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose = pose_msg.pose
        self.odom_pub.publish(odom)

        if self.tf_br is not None:
            self.tf_br.sendTransform(
                (mean_x, mean_y, 0.0),
                (
                    pose_msg.pose.orientation.x,
                    pose_msg.pose.orientation.y,
                    pose_msg.pose.orientation.z,
                    pose_msg.pose.orientation.w,
                ),
                stamp,
                self.base_frame,
                self.map_frame,
            )

    @staticmethod
    def _particle_to_pose(particle):
        pose = Pose()
        pose.position.x = float(particle[0])
        pose.position.y = float(particle[1])
        pose.position.z = 0.0
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, float(particle[2]))
        pose.orientation = Quaternion(*q)
        return pose


def main():
    rospy.init_node("particle_filter_node")
    ParticleFilterNode()
    rospy.spin()


if __name__ == "__main__":
    main()
