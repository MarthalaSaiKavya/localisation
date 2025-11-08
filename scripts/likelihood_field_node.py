#!/usr/bin/env python3
import os

import cv2
import numpy as np
import rospy

from pf_localization.map_utils import load_likelihood_field


class LikelihoodFieldNode:
    def __init__(self):
        pkg_dir = rospy.get_param(
            "~pkg_dir", os.path.expanduser("~/catkin_ws/src/pf_localization")
        )
        map_yaml = rospy.get_param(
            "~map_yaml", os.path.join(pkg_dir, "maps", "house_map.yaml")
        )
        out_png = rospy.get_param(
            "~out_png", os.path.join(pkg_dir, "images", "likelihood_field.png")
        )

        self.field = load_likelihood_field(map_yaml)
        vis = self._create_visualization(self.field.distance_field)
        if not cv2.imwrite(out_png, vis):
            raise RuntimeError(f"Failed to write likelihood field PNG: {out_png}")

        rospy.loginfo("Saved likelihood field preview to %s", out_png)
        rospy.loginfo(
            "Distance field stats — min: %.3f m, max: %.3f m",
            float(np.min(self.field.distance_field)),
            float(np.max(self.field.distance_field)),
        )

    @staticmethod
    def _create_visualization(dist_m):
        vis = dist_m.copy()
        finite = np.isfinite(vis)
        vmax = np.percentile(vis[finite], 99) if finite.any() else 1.0
        vmax = max(vmax, 1e-3)
        vis = np.clip(vis / vmax, 0.0, 1.0)
        vis = (vis * 255).astype(np.uint8)
        return cv2.GaussianBlur(vis, (5, 5), 0)


def main():
    rospy.init_node("likelihood_field_node")
    LikelihoodFieldNode()
    rospy.spin()


if __name__ == "__main__":
    main()
