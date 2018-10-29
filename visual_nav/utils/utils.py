import math
from math import *
import numpy as np
from transforms3d import euler, quaternions, affines
import airsim

def coordinate_transform(image):
    pass

class PinholeCameraForwardProjection():

    def __init__(self,
                 img_x=224,
                 img_y=224,
                 cam_fov=90,
                 shear=0):

        self.res_x = img_x
        self.res_y = img_y
        self.cam_fov = cam_fov
        self.s = shear

    def get_focal_len(self):
        return self.res_x * 0.5 / math.tan(self.cam_fov * 0.5 * pi / 180)

    def make_camera_matrix(self):
        f = self.get_focal_len()
        return np.asarray([[f, 0, self.res_x / 2, 0], [0, f, self.res_y / 2, 0], [0, 0, 1, 0]])

    def make_rotation_matrix(self, quat):
        return quaternions.quat2mat(quat)

    def make_world_to_camera_mat(self, cam_pos, cam_rot):
        # http://ksimek.github.io/2012/08/22/extrinsic/
        rot_mat = quaternions.quat2mat(cam_rot)
        t = - np.matmul(rot_mat.T, cam_pos)
        mat = affines.compose(t, rot_mat.T, [1.0, 1.0, 1.0])
        return mat

    def make_optical_rotation_matrix(self):
        return affines.compose([0, 0, 0], euler.euler2mat(-pi / 2, 0, -pi / 2).T, [1, 1, 1])

    def world_point_to_image(self, cam_pos, cam_rot, pts_w):

        T = self.make_world_to_camera_mat(cam_pos, cam_rot)
        R_opt = self.make_optical_rotation_matrix()  # Point in camera's optical frame
        K = self.make_camera_matrix()

        pts_cam = np.matmul(T, pts_w)
        pts_cam_opt = np.matmul(R_opt, pts_cam)

        if pts_cam_opt[2] < 0:
            return None, "point behind"

        pts_img = np.matmul(K, pts_cam_opt)
        pts_img_in_pixel = [self.res_y - int(pts_img[1] / pts_img[2]), self.res_x - int(pts_img[0] / pts_img[2])]

        if pts_img_in_pixel[0] < -0.5 or pts_img_in_pixel[0] > self.res_x + 0.5 or \
                pts_img_in_pixel[1] < -0.5 or pts_img_in_pixel[1] > self.res_y + 0.5:
            return None, "point out of fov"

        return pts_img_in_pixel, "point within fov"

    def image_to_world(self, image):



if __name__ == '__main__':
    from operator import add

    projector = PinholeCameraForwardProjection()
    # # gridsize = tan(4.5/180 *pi) * 2 * 17.5/100
    # gridsize = 0
    # pts_w = [-40.95000076293945, -5.75, 0, 1]
    # # list( map(add, list1, list2) )
    # # pts_w = list( map(add, [-25.02, 10.159999999999997, 0, 1] , [gridsize/2, gridsize/2, 0,0] ))
    # # cam_pos = [ 2501.9996643066406, -1015.999984741211, -17.499999701976776]
    # # vehicle_pos = [ -25.019996643066406, 10.15999984741211, -0.17499999701976776]
    # vehicle_pos = [-40.95000076293945, -5.75, -0.7]
    # # vehicle_rot = [0.7071067094802856, 0.0, -0.707106790849304, 0.0]
    # vehicle_rot = [0.7071067094802856, -0, -0.7071067094802856, 0]
    # pts_img = projector.world_point_to_image(vehicle_pos, vehicle_rot, pts_w)
    # print(pts_img)

    # use open cv to create point cloud from depth image.
    import math
    import sys

    import airsim
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # file will be saved in PythonClient folder (i.e. same folder as script)
    # point cloud ASCII format, use viewers like CloudCompare http://www.danielgm.net/cc/
    outputFile = "cloud.asc"
    color = (0, 255, 0)
    rgb = "%d %d %d" % color
    projectionMatrix = np.array([[-0.501202762, 0.000000000, 0.000000000, 0.000000000],
                                 [0.000000000, -0.501202762, 0.000000000, 0.000000000],
                                 [0.000000000, 0.000000000, 10.00000000, 100.00000000],
                                 [0.000000000, 0.000000000, -10.0000000, 0.000000000]])


    def printUsage():
        print("Usage: python point_cloud.py [cloud.txt]")


    def savePointCloud(image, fileName):
        f = open(fileName, "w")
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                pt = image[x, y]
                if (math.isinf(pt[0]) or math.isnan(pt[0])):
                    # skip it
                    None
                else:
                    f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2] - 1, rgb))
        f.close()


    for arg in sys.argv[1:]:
        cloud.txt = arg

    client = airsim.MultirotorClient()

    while True:
        rawImage = client.simGetImage("0", airsim.ImageType.DepthPerspective)
        if (rawImage is None):
            print("Camera is not returning image, please check airsim for error messages")
            airsim.wait_key("Press any key to exit")
            sys.exit(0)
        else:
            png = cv2.imdecode(np.frombuffer(rawImage, np.uint8), cv2.IMREAD_UNCHANGED)
            gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Depth image', gray)
            cv2.waitKey(0)
            Image3D = cv2.reprojectImageTo3D(gray, projectionMatrix)
            savePointCloud(Image3D, outputFile)
            print("saved " + outputFile)
            airsim.wait_key("Press any key to exit")
            sys.exit(0)

        key = cv2.waitKey(1) & 0xFF;
        if (key == 27 or key == ord('q') or key == ord('x')):
            break;
