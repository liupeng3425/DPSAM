import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
class PointCloud(object):
    def __init__(self, H=64, W=1024, fov_down=-30, fov_up=10, lidar_position=(1, 0, 2)):
        self.fov_up = fov_up / 180.0 * np.pi  # in rad
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov = self.fov_up - self.fov_down
        self.H = H
        self.W = W
        self.lidar_position = np.asarray(lidar_position)

    def do_range_projection(self, points, semantics):
        # restore points coordinate to original carla's lidar.
        points_carla = points * np.array([1, -1, 1])
        points_carla -= self.lidar_position

        depth = np.linalg.norm(points_carla, 2, axis=1)

        x = points_carla[:, 0]
        y = -points_carla[:, 1]  # carla-coor is left-hand.
        z = points_carla[:, 2]

        yaw = np.arctan2(y, x)
        pitch = np.arcsin(z / depth)

        proj_w = 0.5 * (1.0 - yaw / np.pi)
        proj_h = 1.0 - (pitch + abs(self.fov_down)) / self.fov
        proj_w *= self.W
        proj_h *= self.H

        proj_w = np.floor(proj_w)
        proj_w = np.minimum(self.W - 1, proj_w)
        proj_w = np.maximum(0, proj_w).astype(np.int32)

        proj_h = np.floor(proj_h)
        proj_h = np.minimum(self.H - 1, proj_h)
        proj_h = np.maximum(0, proj_h).astype(np.int32)

        # After sorting by depth from largest to smallest, close point will rewrite distant point in the same pixel.
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        proj_w = proj_w[order]
        proj_h = proj_h[order]
        points = points[order]
        semantics = semantics[order]

        range_depth = np.full((self.H, self.W), -1, dtype=np.float32)
        range_xyz = np.full((self.H, self.W, 3), 0, dtype=np.float32)
        range_sem = np.full((self.H, self.W), 0, dtype=np.uint8)

        # points += self.lidar_position
        # points[:, 1] *= -1

        range_depth[proj_h, proj_w] = depth
        range_xyz[proj_h, proj_w] = points
        range_sem[proj_h, proj_w] = semantics
        return range_depth, range_xyz, range_sem

    # re-projection range-view pcd to original coordinate.
    def restore_pcd_coor(self, range_depth):
        h, w = np.arange(0, self.H), np.arange(0, self.W)
        proj_w, proj_h = np.meshgrid(w, h)
        # valid = range_depth > 0
        proj_w = proj_w.astype(float)[None, None, ...]
        proj_h = proj_h.astype(float)[None, None, ...]
        depth = range_depth

        proj_w /= self.W
        proj_h /= self.H
        pitch = (1.0 - proj_h) * self.fov - abs(self.fov_down)
        yaw = (1.0 - proj_w / 0.5) * np.pi

        z = depth * np.sin(pitch)
        depth_ = depth * np.cos(pitch)
        x = depth_ * np.cos(yaw)
        y = depth_ * np.sin(yaw)

        points = np.concatenate([x[..., None], -y[..., None], z[..., None]], axis=-1)
        points += self.lidar_position.reshape((1, 1, 1, 1, -1))
        points *= np.array([1, -1, 1]).reshape((1, 1, 1, 1, -1))
        return np.concatenate([points, depth[..., None]], axis=-1)




def align_img(img: np.ndarray, img_size: int = 64) -> np.ndarray:
    """Aligns the image to the center.
    Args:
        img (np.ndarray): Image to align.
        img_size (int, optional): Image resizing size. Defaults to 64.
    Returns:
        np.ndarray: Aligned image.
    """    
    if img.sum() <= 10000:
        y_top = 0
        y_btm = img.shape[0]
    else:
        # Get the upper and lower points
        # img.sum
        y_sum = img.sum(axis=2).sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)

    img = img[y_top: y_btm, :,:]

    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)
    
    # Get the median of the x-axis and take it as the person's x-center.
    x_csum = img.sum(axis=2).sum(axis=0).cumsum()
    x_center = img.shape[1] // 2
    for idx, csum in enumerate(x_csum):
        if csum > img.sum() / 2:
            x_center = idx
            break

    # if not x_center:
    #     logging.warning(f'{img_file} has no center.')
    #     continue

    # Get the left and right points
    half_width = img_size // 2
    left = x_center - half_width
    right = x_center + half_width
    if left <= 0 or right >= img.shape[1]:
        left += half_width
        right += half_width
        # _ = np.zeros((img.shape[0], half_width,3))
        # img = np.concatenate([_, img, _], axis=1)
    
    img = img[:, left: right,:].astype('uint8')
    return img



def lidar_to_2d_front_view(points,
                           v_res,
                           h_res,
                           v_fov,
                           val="depth",
                           cmap="jet",
                           saveto=None,
                           y_fudge=0.0
                           ):
    """ Takes points in 3D space from LIDAR data and projects them to a 2D
        "front view" image, and saves that image.

    Args:
        points: (np array)
            The numpy array containing the lidar points.
            The shape should be Nx4
            - Where N is the number of points, and
            - each point is specified by 4 values (x, y, z, reflectance)
        v_res: (float)
            vertical resolution of the lidar sensor used.
        h_res: (float)
            horizontal resolution of the lidar sensor used.
        v_fov: (tuple of two floats)
            (minimum_negative_angle, max_positive_angle)
        val: (str)
            What value to use to encode the points that get plotted.
            One of {"depth", "height", "reflectance"}
        cmap: (str)
            Color map to use to color code the `val` values.
            NOTE: Must be a value accepted by matplotlib's scatter function
            Examples: "jet", "gray"
        saveto: (str or None)
            If a string is provided, it saves the image as this filename.
            If None, then it just shows the image.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical range do not match the actual data.

            For a Velodyne HDL 64E, set this value to 5.
    """

    # DUMMY PROOFING
    assert len(v_fov) ==2, "v_fov must be list/tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"
    assert val in {"depth", "height", "reflectance"}, \
        'val must be one of {"depth", "height", "reflectance"}'


    x_lidar = - points[:, 0]
    y_lidar = - points[:, 1]
    z_lidar = points[:, 2]
    # Distance relative to origin when looked from top
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]

    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)

    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad
    y_img = np.arctan2(z_lidar, d_lidar)/ v_res_rad

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
    x_img -= x_min              # Shift
    x_max = 360.0 / h_res       # Theoretical max x value after shifting

    y_min = v_fov[0] / v_res    # theoretical min y value based on sensor specs
    y_img -= y_min              # Shift
    y_max = v_fov_total / v_res # Theoretical max x value after shifting

    y_max += y_fudge            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.

    # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
    if val == "reflectance":
        pass
    elif val == "height":
        pixel_values = z_lidar
    else:
        pixel_values = -d_lidar
        # pixel_values = 'w'

    # PLOT THE IMAGE
    cmap = "jet"            # Color map to use
    dpi = 100               # Image resolution
    fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
    ax.scatter(x_img,y_img, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
    ax.set_facecolor((0, 0, 0)) # Set regions with no points to black
    ax.axis('scaled')              # {equal, scaled}
    ax.xaxis.set_visible(False)    # Do not draw axis tick marks
    ax.yaxis.set_visible(False)    # Do not draw axis tick marks
    plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
    plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV

    saveto = saveto.replace('.pcd','.png')
    fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    img = cv2.imread(saveto)
    img = align_img(img)

    aligned_path = saveto.replace('offline','aligned')
    os.makedirs(os.path.dirname(aligned_path), exist_ok=True)
    cv2.imwrite(aligned_path, img)
    # fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
    # ax.scatter(x_img,y_img, s=1, c='white', linewidths=0, alpha=1)
    # ax.set_facecolor((0, 0, 0)) # Set regions with no points to black
    # ax.axis('scaled')              # {equal, scaled}
    # ax.xaxis.set_visible(False)    # Do not draw axis tick marks
    # ax.yaxis.set_visible(False)    # Do not draw axis tick marks
    # plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
    # plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV

    # fig.savefig(saveto.replace('depth','sils'), dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    # plt.close()


# 读取二进制文件
def read_bin_file(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
    data = np.frombuffer(content, dtype=np.float32).reshape(-1, 4)  # 每个点由(x, y, z, intensity)四个值组成
    return data       

class VeloToImage:
    def __init__(self) -> None:
        self.dict_velo_img = {}
        self.cams = {}

    def projectVeloToImage(self, velo_path, out_path, cam_id=0):
        from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
        from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
        from PIL import Image
        import matplotlib.pyplot as plt

        # if 'KITTI360_DATASET' in os.environ:
        #     kitti360Path = os.environ['KITTI360_DATASET']
        # else:
        #     kitti360Path = os.path.join(os.path.dirname(
        #                             os.path.realpath(__file__)), '..', '..')
        kitti360Path = '/data1/liupeng/data/kitti360/down_2d_train_val/KITTI-360'
        save_path = os.path.join(out_path, os.path.dirname(velo_path))
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        seq = int(velo_path.split('/2013_05_28_drive_')[-1][:4])
        frame = int(velo_path.split('/')[-1][:-4])
        
        sequence = '2013_05_28_drive_%04d_sync'%seq
        if f'{sequence}_{cam_id}' not in self.cams:
            # perspective camera
            if cam_id in [0,1]:
                camera = CameraPerspective(kitti360Path, sequence, cam_id)
            # fisheye camera
            elif cam_id in [2,3]:
                camera = CameraFisheye(kitti360Path, sequence, cam_id)
            else:
                raise RuntimeError('Unknown camera ID!')
            self.cams[f'{sequence}_{cam_id}'] = camera
        else:
            camera = self.cams[f'{sequence}_{cam_id}']

        if sequence not in self.dict_velo_img:

            # cam_0 to velo
            fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
            TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)

            # all cameras to system center 
            fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
            TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)

            # velodyne to all cameras
            TrVeloToCam = {}
            for k, v in TrCamToPose.items():
                # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
                TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
                TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
                # Tr(velo -> cam_k)
                TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)
            
            # take the rectification into account for perspective cameras
            if cam_id==0 or cam_id == 1:
                TrVeloToRect = np.matmul(camera.R_rect, TrVeloToCam['image_%02d' % cam_id])
            else:
                TrVeloToRect = TrVeloToCam['image_%02d' % cam_id]
            self.dict_velo_img[sequence] = TrVeloToRect
        else:
            TrVeloToRect = self.dict_velo_img[sequence]

        # color map for visualizing depth map
        cm = plt.get_cmap('jet')

        # visualize a set of frame
        # for each frame, load the raw 3D scan and project to image plane
        # for frame in range(0, 1000, 50):
        # points = velo.loadVelodyneData(frame)
        pcd_path = os.path.join(kitti360Path, velo_path)
        points = read_bin_file(pcd_path)

        # points[:,3] = 1

        pointsCam = np.matmul(TrVeloToRect, points.T).T
        pointsCam = pointsCam[:,:3]
        # project to image space
        u,v, depth_real= camera.cam2image(pointsCam.T)
        depth =(points[:,3]*100).astype(int)
        u = u.astype(int)
        v = v.astype(int)

        # prepare depth map for visualization
        depthMap = np.zeros((camera.height, camera.width))
        depthImage = np.zeros((camera.height, camera.width, 3))
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)
        # visualize points within 30 meters
        mask = np.logical_and(np.logical_and(mask, depth_real>0), depth_real<40)
        depthMap[v[mask],u[mask]] = depth[mask]
        layout = (2,1) if cam_id in [0,1] else (1,2)
        sub_dir = 'data_rect' if cam_id in [0,1] else 'data_rgb'
        # fig, axs = plt.subplots(*layout, figsize=(18,12))

        # load RGB image for visualization
        imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, sub_dir, '%010d.png' % frame)
        if not os.path.isfile(imagePath):
            raise RuntimeError('Image file %s does not exist!' % imagePath)

        colorImage = np.array(Image.open(imagePath)) / 255.
        # depthImage = cm(depthMap/depthMap.max())[...,:3]
        # depthImage = np.tile(depthMap, (3,1,1))
        depthImage = np.repeat(depthMap[:, :, np.newaxis], 3, axis=2)
        colorImage[depthMap>0] = depthImage[depthMap>0]
        cv2.imwrite(os.path.join(out_path, velo_path.replace('.bin', '.png')), depthImage)

        # axs[0].imshow(depthMap, cmap='jet', interpolation='none')
        # axs[0].imshow(depthImage)
        # axs[0].title.set_text('Projected Depth')
        # axs[0].axis('off')
        # axs[1].imshow(colorImage)
        # axs[1].title.set_text('Projected Depth Overlaid on Image')
        # axs[1].axis('off')
        # plt.suptitle('Sequence %04d, Camera %02d, Frame %010d' % (seq, cam_id, frame))
        
        # fig.savefig(os.path.join(out_path, velo_path.replace('.bin', '.png')))
        # plt.show()