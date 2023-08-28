import onnxruntime
import numpy as np
import cv2
import time
import os
from PIL import Image,ImageDraw,ImageFont
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新', '学', '警', '港', '澳', '挂', '使', '领', '民', '深',
         '危', '险', '空',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
# 调色板
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])
# 17个关键点连接顺序
skeleton = [[1, 2], [2, 3], [3, 4]]
# 骨架颜色
pose_limb_color = palette[[9, 7, 0, 16]]
# 关键点颜色
pose_kpt_color = palette[[16, 0, 9, 8]]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    '''  调整图像大小和两边灰条填充  '''
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # 缩放比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 只进行下采样 因为上采样会让图片模糊
    if not scaleup:
        r = min(r, 1.0)
    # 计算pad长宽
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 保证缩放后图像比例不变
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # 在较小边的两侧进行pad, 而不是在一侧pad
    dw /= 2
    dh /= 2
    # 将原图resize到new_unpad（长边相同，比例相同的新图）
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 计算上下两侧的padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # 计算左右两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 添加灰条
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

def pre_process(img):
    # 归一化 调整通道为（1，3，640，640）
    img = img / 255.
    img = np.transpose(img, (2, 0, 1))
    data = np.expand_dims(img, axis=0)
    return data

def xywh2xyxy(x):
    ''' 中心坐标、w、h ------>>> 左上点，右下点 '''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# nms算法
def nms(dets, iou_thresh):
    # dets: N * M, N是bbox的个数，M的前4位是对应的 左上点，右下点
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标
    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)
        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= iou_thresh)[0]
        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    output = []
    for i in keep:
        output.append(dets[i].tolist())
    return np.array(output)

def xyxy2xywh(a):

    ''' 左上点 右下点 ------>>> 左上点 宽 高 '''
    b = np.copy(a)
    # y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    # y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    try:
        b[:, 2] = a[:, 2] - a[:, 0]  # w
        b[:, 3] = a[:, 3] - a[:, 1]  # h
    except IndexError:
        return b
    return b

def scale_boxes(img1_shape, boxes, img0_shape):
    '''   将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param boxes:  预测的box信息
    :param img0_shape: 原始图像尺度
    '''
    # 将检测框(x y w h)从img1_shape(预测图) 缩放到 img0_shape(原图)
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    boxes[:, 0] -= pad[0]
    boxes[:, 1] -= pad[1]
    boxes[:, :4] /= gain  # 检测框坐标点还原到原图上
    # num_kpts = boxes.shape[1] // 3   # 56 // 3 = 18
    # print(num_kpts)
    # 4个特征点，索引7-14（8个）是下标
    boxes[:, 7] = (boxes[:, 7] - pad[0]) / gain
    boxes[:, 8] = (boxes[:, 8] - pad[1]) / gain    
    boxes[:, 9] = (boxes[:, 9] - pad[0]) / gain
    boxes[:, 10] = (boxes[:, 10] - pad[1]) / gain  
    boxes[:, 11] = (boxes[:, 11] - pad[0]) / gain
    boxes[:, 12] = (boxes[:, 12] - pad[1]) / gain  
    boxes[:, 13] = (boxes[:, 13] - pad[0]) / gain
    boxes[:, 14] = (boxes[:, 14] - pad[1]) / gain   

    clip_boxes(boxes, img0_shape)
    return boxes
def clip_boxes(boxes, shape):
    # 进行一个边界截断，以免溢出
    # 并且将检测框的坐标（左上角x，左上角y，宽度，高度）--->>>（左上角x，左上角y，右下角x，右下角y）
    top_left_x = boxes[:, 0].clip(0, shape[1])
    top_left_y = boxes[:, 1].clip(0, shape[0])
    bottom_right_x = (boxes[:, 0] + boxes[:, 2]).clip(0, shape[1])
    bottom_right_y = (boxes[:, 1] + boxes[:, 3]).clip(0, shape[0])
    boxes[:, 0] = top_left_x      #左上
    boxes[:, 1] = top_left_y
    boxes[:, 2] = bottom_right_x  #右下
    boxes[:, 3] = bottom_right_y

def plot_skeleton_kpts(im, kpts, steps=2):
    num_kpts = len(kpts) // steps  # 51 / 3 =17 
    #print(num_kpts)
    key_points = []
    # num_kpts = 4
    # 画点
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        # conf = kpts[steps * kid + 2]
        # if conf > 0.5:   # 关键点的置信度必须大于 0.5
        cv2.circle(im, (int(x_coord), int(y_coord)), 10, (int(r), int(g), int(b)), -1)
        key_points.append((x_coord,y_coord))
    return key_points
    # # 画骨架
    # for sk_id, sk in enumerate(skeleton):
    #     r, g, b = pose_limb_color[sk_id]
    #     pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
    #     pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
    #     conf1 = kpts[(sk[0]-1)*steps+2]
    #     conf2 = kpts[(sk[1]-1)*steps+2]
    #     if conf1 >0.5 and conf2 >0.5:  # 对于肢体，相连的两个关键点置信度 必须同时大于 0.5
    #         cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
#图像前处理
def preprocess_image(img, img_size):
    
    img = cv2.resize(img, img_size)
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return img


def Greedy_Decode(prebs):
    # TestNet = Net.eval()
    # greedy decode
    index =np.argmax(prebs[0],axis=1)
    #print(index)
    no_repeat_blank_label = list()
    preb_label=index[0]
    pre_c = preb_label[0]
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in preb_label: # dropout repeate label and blank label
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    # preb_labels.append(no_repeat_blank_label) 
    lb = ""
    for k in no_repeat_blank_label:
        lb += CHARS[k]
    #print(f"Prediction: {lb}")
    return lb
#模型推理
def onnx_runtime(image,onnx_path):
    img_size = [94,24]

    # 预处理图像

    preprocessed_image = preprocess_image(image, img_size)
    sess = onnxruntime.InferenceSession(onnx_path)  #加载onnx模型

    # 调整输入的维度形状以匹配模型的输入要求

    input_name = sess.get_inputs()[0].name
    imgdata = np.expand_dims(preprocessed_image, axis=0)
    output_name = sess.get_outputs()[0].name  
    pred_onnx = sess.run([output_name], {input_name: imgdata}) #模型推理

    #解码

    lb = Greedy_Decode(pred_onnx)
    #image
    return lb
def order_points(pts):
    # 初始化坐标点
    rect = np.zeros((4,2), dtype='float32')
    
    # 获取左上角和右下角坐标点
    s = pts.sum(axis=1) # 每行像素值进行相加；若axis=0，每列像素值相加
    rect[0] = pts[np.argmin(s)] # top_left,返回s首个最小值索引，eg.[1,0,2,0],返回值为1
    rect[2] = pts[np.argmax(s)] # bottom_left,返回s首个最大值索引，eg.[1,0,2,0],返回值为2
    
    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis=1) # 第i+1列减第i列
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def process_image(image,keypoints):
    #     # Predict with the model
    #     results = model(image_path)
    #     keypoints = results[0].keypoints  # Get keypoints for the first detection
    #     keypoints = keypoints[0]
        
    #     # Convert the CPU tensor to a NumPy array
    #     keypoints_cpu = keypoints.cpu()
    #     keypoints_array = keypoints_cpu.numpy()
    #     corner_points = keypoints_array
    #     feature_points_data = np.array(corner_points)
    #     feature_points = [tuple(point) for point in feature_points_data]
    rect = order_points(keypoints)
    
    # width = x2 - x1
    # height = y2 - y1
    # Define the output width and height for the warped plate
    output_width, output_height = 94, 24
    output_points = np.array([[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, output_points)
    # M = cv2.getPerspectiveTransform(corner_points, output_points)

    # Perform the affine transformation
    warped_plate = cv2.warpPerspective(image, M, (output_width, output_height))
    return warped_plate

def cv2ImgAddText(img,text,left,top,textColor,textSize):
    if(isinstance(img,np.ndarray)):
        img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    draw =ImageDraw.Draw(img)
    ttf="/home/can.dong/LPRNet_Pytorch/data/NotoSansCJK-Regular.ttc"     #linux中的中文字体格式一般在/usr/share/fonts/opentype/noto下
    fontStyle=ImageFont.truetype(
    ttf,textSize,encoding="utf-8")
    draw.text((left,top),text,textColor,font=fontStyle)
    return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

class Keypoint():
    def __init__(self,modelpath):
        # self.session = onnxruntime.InferenceSession(modelpath, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.session = onnxruntime.InferenceSession(modelpath, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name
    def inference(self,image):
        img = letterbox(image)
        data = pre_process(img)
        # 预测输出float32[1, 15, 8400]
        #15 = 4 + 1 + 2 + 8 
        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        # print(pred.shape)
        # [15, 8400]
        pred = pred[0]
        # [8400,15]
        pred = np.transpose(pred, (1, 0))
        # 置信度阈值过滤
        conf = 0.003
        pred = pred[pred[:, 4] > conf]
        # 中心宽高转左上点，右下点
        try:
            bboxs = xywh2xyxy(pred)
            # NMS处理
            bboxs = nms(bboxs, iou_thresh=0.5)
            # 坐标从左上点，右下点 到 左上点，宽，高.
            bboxs = np.array(bboxs)
            bboxs = xyxy2xywh(bboxs)
            # 坐标点还原到原图
            bboxs = scale_boxes(img.shape, bboxs, image.shape)
        except:
            return image, [] ,()
        # 画框 画点 画骨架
        for box in bboxs:
            # 依次为 检测框（左上点，右下点）、置信度、17个关键点
            det_bbox, det_scores, kpts = box[0:4], box[4], box[7:]
            #print(kpts.shape)
            # 画框
            x1, y1, x2, y2 = int(det_bbox[0]), int(det_bbox[1]), int(det_bbox[2]), int(det_bbox[3])
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            print("Width:", width)
            print("Height:", height)
            cv2.rectangle(image, (int(det_bbox[0]), int(det_bbox[1])), (int(det_bbox[2]), int(det_bbox[3])),
                                (0, 0, 255), 2)
            # 人体检测置信度
            if int(det_bbox[1]) < 30 :
                cv2.putText(image, "conf:{:.2f}".format(det_scores), (int(det_bbox[0]) + 5, int(det_bbox[1]) +25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
            else:
                cv2.putText(image, "conf:{:.2f}".format(det_scores), (int(det_bbox[0]) + 5, int(det_bbox[1]) - 5),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
            # 画点 连线
            key_points = plot_skeleton_kpts(image, kpts)
            pos = (int(det_bbox[0]) + 5, int(det_bbox[1]) +25)
        return image,key_points,pos

if __name__ == '__main__':
    modelpath = r'/home/can.dong/ultralytics/runs/pose/train55/weights/best.onnx'
    # 实例化模型
    keydet = Keypoint(modelpath)
    License_modelpath = '/home/can.dong/LPRNet_Pytorch/best_0807.onnx'
    # 两种模式 1为图片预测，并显示结果图片；2为摄像头检测，并实时显示FPS;3为读入视频流
    img_size = [94,24]

    # 预处理图像

    
    sess = onnxruntime.InferenceSession(License_modelpath)  #加载onnx模型
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name  
    mode = 3
    if mode == 1:
        # 输入图片路径
        image = cv2.imread('/home/can.dong/demo1/6.jpg')
        start = time.time()
        image,key,pos = keydet.inference(image)
        end = time.time()
        det_time = (end - start) * 1000
        print("推理时间为：{:.2f} ms".format(det_time))
        print("图片完成检测")
        # cv2.namedWindow("keypoint", cv2.WINDOW_NORMAL)
        # cv2.imshow("keypoint", image)
        cv2.imwrite('/home/can.dong/new/ultralytics/onnx.jpg',image)
    elif mode == 2:
        # 摄像头人体关键点检测
        cap = cv2.VideoCapture(0)
        # 返回当前时间
        start_time = time.time()
        counter = 0
        while True:
            # 从摄像头中读取一帧图像
            ret, frame = cap.read()
            image  = keydet.inference(frame)
            counter += 1  # 计算帧数
            # 实时显示帧数
            if (time.time() - start_time) != 0:
                cv2.putText(image, "FPS:{0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                # 显示图像
                cv2.imshow('keypoint', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
    elif mode == 3:
        video_path = "/home/can.dong/demo/vehicle.mp4"  # 视频文件路径
        # 读取视频文件
        cap = cv2.VideoCapture(video_path)
        # 获取视频的帧速率和尺寸
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 创建VideoWriter对象，用于保存新的视频
        output = cv2.VideoWriter('output_video_080_test1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        counter = 0
        while True:
            # 从摄像头中读取一帧图像
            ret, frame = cap.read()
            if not ret:
                break
            image,key_points,pos  = keydet.inference(frame)

            key_points = np.array(key_points)
            if len(key_points) > 0:
                wrap_img = process_image(image,key_points)
                #to do：提取特征点做仿射变换
                preprocessed_image = preprocess_image(wrap_img, img_size)
                imgdata = np.expand_dims(preprocessed_image, axis=0)
                pred_onnx = sess.run([output_name], {input_name: imgdata}) #模型推理

                #解码

                lb = Greedy_Decode(pred_onnx)

                #lb = onnx_runtime(wrap_img,License_modelpath)
                #把lb标注在图片上？？
                image=cv2ImgAddText(image,lb,pos[0]-10,pos[1]+60,(153, 255, 153),40)#(x,y),往下+y，往右-x
                cv2.imwrite('/home/can.dong/demo/test2/'+str(counter)+'.jpg',image)
            counter += 1  # 计算帧数
             # 写入视频文件
                 # 保存处理后的帧到新的视频
            output.write(image)
    else:
        print("\033[1;91m 输入错误，请检查mode的赋值 \033[0m")
