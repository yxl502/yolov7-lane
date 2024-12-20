import numpy as np
import cv2
from mymodel import YoloV7
from myfunc import weighted_img, draw_lines, hough_lines


if __name__ == '__main__':
    print("[INFO] 开始YoloV7模型加载")
    # YOLOV5模型配置文件(YAML格式)的路径 yolov5_yaml_path
    model = YoloV7(yolov7_yaml_path='cfg/training/yolov7s-lane.yaml')
    print("[INFO] 完成YoloV7模型加载")
    # low = 200 #Canny low
    low = 183 #Canny low
    # high = 300 #Canny high
    high = 213 #Canny high
    rho = 1  # 霍夫像素单位
    theta = np.pi / 360  # 霍夫角度移动步长
    hof_threshold = 20  # 霍夫平面累加阈值threshold
    min_line_len = 10  # 线段最小长度
    max_line_gap = 20  # 最大允许断裂长度
    index = 0 #图片索引
    while True:
        index = index + 1
        # 一张一张图片进行检测按index进行索引
        path = '2.png'
        # path = '4.jpg'
        path_output = r"./out/" + str(index) + ".jpg"
        color_image = cv2.imread(path)
        lane_img=color_image.copy()
        edges = cv2.Canny(lane_img, low, high)
        cv2.imshow('edges', edges)
        cv2.waitKey(0)

        mask = np.zeros_like(edges)
        # vertices = np.array( [[(554, 463), (733, 464), (1112, 654), (298, 671)]],dtype=np.int32)#素材2的ROI
        vertices = np.array( [[(17, 380), (145, 176), (324, 179), (304, 376)]],dtype=np.int32)#素材1的ROI
        cv2.fillPoly(mask, vertices, 255)#绿色蒙板绘制
        masked_edges = cv2.bitwise_and(edges, mask)  # 按位与
        line_image = np.zeros_like(lane_img)
        # 绘制车道线线段
        lines = hough_lines(masked_edges, rho, theta, hof_threshold, min_line_len, max_line_gap)
        draw_lines(line_image, lines, thickness=10)
        # YoloV5 目标检测
        canvas, class_id_list, xyxy_list, conf_list = model.detect(color_image)
        if xyxy_list:
            for i in range(len(xyxy_list)):
                ux = int((xyxy_list[i][0] + xyxy_list[i][2]) / 2)  # 计算像素坐标系的x
                uy = int((xyxy_list[i][1] + xyxy_list[i][3]) / 2)  # 计算像素坐标系的y
                cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
                cv2.putText(canvas, str([ux, uy]), (ux + 20, uy + 10), 0, 1,
                            [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)  # 标出坐标
        canvas=weighted_img(canvas, line_image)
        # 可视化部分
        # cv2.namedWindow("raw", 0)
        # cv2.imshow('raw',color_image)
        # cv2.namedWindow("line", 0)
        # cv2.imshow('line',line_image)
        cv2.namedWindow('detection', 0)
        cv2.imshow('detection', canvas)
        cv2.imwrite(path_output, canvas)#图片保存
        key = cv2.waitKey()
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
