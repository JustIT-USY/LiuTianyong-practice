import pyrealsense2 as rs
import numpy as np
import cv2
import time
import open3d as o3d
import os

view_ind = 0
breakLoopFlag = 0
backgroundColorFlag = 1


# 保存深度图、RGB图像 和 点云图
def saveCurrentRGBD(vis):
    global view_ind
    if not os.path.exists('./output/'):
        os.makedirs('./output')
    cv2.imwrite('./output/depth/depth_' + str(view_ind) + '.png', depth_image)
    cv2.imwrite('./output/color/color_' + str(view_ind) + '.png', color_image1)
    o3d.io.write_point_cloud('./output/pointcloud/pointcloud_' + str(view_ind) + '.pcd', pcd)
    print('No.' + str(view_ind) + ' shot is saved.')
    view_ind += 1
    return False


# 退出
def breakLoop(vis):
    global breakLoopFlag
    breakLoopFlag += 1
    return False


# 改变点云图背景
def change_background_color(vis):
    global backgroundColorFlag
    opt = vis.get_render_option()
    if backgroundColorFlag:
        opt.background_color = np.asarray([0, 0, 0])
        backgroundColorFlag = 0
    else:
        opt.background_color = np.asarray([1, 1, 1])
        backgroundColorFlag = 1
    # background_color ~=backgroundColorFlag
    return False


if __name__ == "__main__":
    '''
    要将深度图像对齐到另一个深度图像，请将align_to参数设置为另一个流类型。
    要将非深度图像对齐到深度图像，请将align_to参数设置为RS2_STREAM_DEPTH。
    相机校准和帧的流类型是根据传递给process()的第一个有效帧集动态确定的。
    '''
    align = rs.align(rs.stream.color)

    '''
    该配置允许管道用户为管道流请求过滤器以及设备选择和配置。
    这是管道创建中的一个可选步骤，因为管道在内部解析其流设备。
    Config为它的用户提供了一种设置过滤器和测试是否与设备的管道要求没有冲突的方法。
    它还允许用户为配置过滤器和管道找到一个匹配的设备，以便显式地选择一个设备，并在流开始之前修改它的控件。
    '''
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)
    '''
    该管道简化了用户与设备和计算机视觉处理模块的交互。
    类抽象了摄像机配置和流，以及视觉模块的触发和线程。
    它让应用程序专注于模块的计算机视觉输出，或设备输出数据。
    该流水线可以管理计算机视觉模块，这些模块被实现为一个处理块。
    管道是处理块接口的使用者，而应用程序使用计算机视觉接口。
    '''
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # 使用摄像头
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx,
                                                                 intr.ppy)
    # print(type(pinhole_camera_intrinsic))

    geometrie_added = False
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Pointcloud")
    pointcloud = o3d.geometry.PointCloud()

    vis.register_key_callback(ord("S"), saveCurrentRGBD)
    vis.register_key_callback(ord("Q"), breakLoop)
    vis.register_key_callback(ord("K"), change_background_color)

    try:
        while True:
            pointcloud.clear()

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_frame = aligned_frames.get_depth_frame()

            depth_frame = rs.decimation_filter(1).process(depth_frame)
            depth_frame = rs.disparity_transform(True).process(depth_frame)
            depth_frame = rs.spatial_filter().process(depth_frame)
            depth_frame = rs.temporal_filter().process(depth_frame)
            depth_frame = rs.disparity_transform(False).process(depth_frame)
            depth_frame = rs.hole_filling_filter().process(depth_frame)

            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            depth_image = np.asanyarray(depth_frame.get_data())
            print(depth_image)
            color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('color image', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            cv2.namedWindow('depth image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('depth image', depth_color_image)

            depth = o3d.geometry.Image(depth_image)
            color = o3d.geometry.Image(color_image)

            rgbd = o3d.geometry.create_rgbd_image_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
            pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # pcd = voxel_down_sample(pcd, voxel_size = 0.003)

            pointcloud += pcd

            if not geometrie_added:
                vis.add_geometry(pointcloud)
                geometrie_added = True

            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()

            key = cv2.waitKey(1)

            if key & 0xFF == ord('S'):
                if not os.path.exists('./output/'):
                    os.makedirs('./output')
                cv2.imwrite('./output/depth/depth_' + str(view_ind) + '.png', depth_image)
                cv2.imwrite('./output/color/color_' + str(view_ind) + '.png', color_image1)
                o3d.write_point_cloud('./output/pointcloud/pointcloud_' + str(view_ind) + '.pcd', pcd)
                print('No.' + str(view_ind) + ' shot is saved.')
                view_ind += 1

            elif key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                vis.destroy_window()

                break

            if breakLoopFlag:
                cv2.destroyAllWindows()
                vis.destroy_window()
                break

    finally:
        pipeline.stop()
