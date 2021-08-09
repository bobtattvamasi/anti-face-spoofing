import argparse
import datetime

import cv2

import logging
import os
import subprocess
import threading
import time

import numpy as np
    
from FaceDetector import BlazeFaceDetector
from face_matcher import Matcher

_MODELS = '/home/pi/pycharm_ssh/models'
_FACE_DETECTOR = 'face_detection_front.tflite'
_ANCHORS_NAME = 'anchors.npy'
_KNOWNPERSON = 'db_rus.pickle'

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)


def start_stream(path=-1, frame_w=800, frame_h=600):
    # init and setup webcam
    stream = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if stream.isOpened():
        stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        stream.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
        stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return stream


def screen_off():
    _ = subprocess.Popen('xset -display :0 s blank', shell=True)
    _ = subprocess.Popen('xset -display :0 s reset', shell=True)
    _ = subprocess.Popen('xset -display :0 s activate', shell=True)
    _ = subprocess.Popen('xset -display :0 s reset', shell=True)


def screen_on():
    _ = subprocess.Popen('xset -display :0 s noblank', shell=True)
    _ = subprocess.Popen('xset -display :0 s reset', shell=True)
    _ = subprocess.Popen('xset -display :0 s activate', shell=True)
    _ = subprocess.Popen('xset -display :0 s reset', shell=True)


def generate_subtitles(vehicle_state, face_on_frame):
    text_message = []
    if vehicle_state.get_param('auth') is None and face_on_frame:
        text_message.append('Здравствуйте')
        text_message.append('Посмотрите в камеру')
    if vehicle_state.get_param('auth') is not None:
        if not vehicle_state.get_param('auth'):
            text_message.append('Идентификация не пройдена')
            text_message.append('Вам запрещено управление ТС')
        elif vehicle_state.get_param('auth') and vehicle_state.get_param('drunk') is None:
            if not vehicle_state.get_param('alco_test_start'):
                text_message.append('{}'.format(vehicle_state.get_param('name')))
                text_message.append('Направьте алкотестер на камеру')
            else:
                text_message.append('{}'.format(vehicle_state.get_param('name')))
                text_message.append('Удерживаете алкотестер и сделайте выдох')
        elif vehicle_state.get_param('auth') and vehicle_state.get_param('drunk'):
            text_message.append('Обнаружены пары этанола!')
            text_message.append('Вам запрещено управление ТС')
        elif vehicle_state.get_param('auth') and not vehicle_state.get_param('drunk'):
            text_message.append('Приятного пути!')

    return text_message


def iou_batch(bbox_frame, face_bbox):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    face_bbox = np.expand_dims(face_bbox, 0)
    bbox_frame = np.expand_dims(bbox_frame, 0)

    xx1 = np.maximum(bbox_frame[..., 0], face_bbox[..., 0])
    yy1 = np.maximum(bbox_frame[..., 1], face_bbox[..., 1])
    xx2 = np.minimum(bbox_frame[..., 2], face_bbox[..., 2])
    yy2 = np.minimum(bbox_frame[..., 3], face_bbox[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bbox_frame[..., 2] - bbox_frame[..., 0]) * (bbox_frame[..., 3] - bbox_frame[..., 1])
              + (face_bbox[..., 2] - face_bbox[..., 0]) * (face_bbox[..., 3] - face_bbox[..., 1]) - wh)
    return (o)


def calc_iou(bbox_frame, face_bbox):
    """
    This function takes the predicted bounding box and ground truth bounding box and
    return the IoU ratio
    """
    x_tl_gt, y_tl_gt, x_br_gt, y_br_gt = bbox_frame
    x_tl_p, y_tl_p, x_br_p, y_br_p = face_bbox

    if (x_tl_gt > x_br_gt) or (y_tl_gt > y_br_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_tl_p > x_br_p) or (y_tl_p > y_br_p):
        raise AssertionError("Predicted Bounding Box is not correct", x_tl_p, x_br_p, y_tl_p,
                             y_br_gt)

    # if the GT bbox and predcited BBox do not overlap then iou=0
    if x_br_gt < x_tl_p or y_br_gt < y_tl_p or x_tl_gt > x_br_p or y_tl_gt > y_br_p:
        return 0.0

    bbox_frame_area = (x_br_gt - x_tl_gt + 1) * (y_br_gt - y_tl_gt + 1)
    face_bbox_area = (x_br_p - x_tl_p + 1) * (y_br_p - y_tl_p + 1)

    x_top_left = np.max([x_tl_gt, x_tl_p])
    y_top_left = np.max([y_tl_gt, y_tl_p])
    x_bottom_right = np.min([x_br_gt, x_br_p])
    y_bottom_right = np.min([y_br_gt, y_br_p])

    intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

    union_area = (bbox_frame_area + face_bbox_area - intersection_area)

    return intersection_area / face_bbox_area


def draw_frame(orig_img, bbox_frame, line_thickness=5, line_color=(0, 255, 0)):
    cv2.line(orig_img, (bbox_frame[0], bbox_frame[1]),
             (bbox_frame[0], bbox_frame[1] + int(abs((bbox_frame[1] - bbox_frame[3])) * 0.2)), line_color,
             thickness=line_thickness)

    cv2.line(orig_img, (bbox_frame[0], bbox_frame[1]),
             (bbox_frame[0] + int(abs((bbox_frame[0] - bbox_frame[2])) * 0.2), bbox_frame[1]), line_color,
             thickness=line_thickness)

    cv2.line(orig_img, (bbox_frame[0], bbox_frame[3]),
             (bbox_frame[0], bbox_frame[3] - int(abs((bbox_frame[1] - bbox_frame[3])) * 0.2)), line_color,
             thickness=line_thickness)

    cv2.line(orig_img, (bbox_frame[0], bbox_frame[3]),
             (bbox_frame[0] + int(abs((bbox_frame[0] - bbox_frame[2])) * 0.2), bbox_frame[3]), line_color,
             thickness=line_thickness)

    cv2.line(orig_img, (bbox_frame[2], bbox_frame[3]),
             (bbox_frame[2], bbox_frame[3] - int(abs((bbox_frame[1] - bbox_frame[3])) * 0.2)), line_color,
             thickness=line_thickness)

    cv2.line(orig_img, (bbox_frame[2], bbox_frame[3]),
             (bbox_frame[2] - int(abs((bbox_frame[0] - bbox_frame[2])) * 0.2), bbox_frame[3]), line_color,
             thickness=line_thickness)

    cv2.line(orig_img, (bbox_frame[2], bbox_frame[1]),
             (bbox_frame[2], bbox_frame[1] + int(abs((bbox_frame[1] - bbox_frame[3]) * 0.2))), line_color,
             thickness=line_thickness)

    cv2.line(orig_img, (bbox_frame[2], bbox_frame[1]),
             (bbox_frame[2] - int(abs((bbox_frame[0] - bbox_frame[2])) * 0.2), bbox_frame[1]), line_color,
             thickness=line_thickness)


def main(vehicle_state, serverSandler, matcher):
    logging.info('INIT')

    lock_in = 37
    lock_out = 38
    GPIO.setup(lock_in, GPIO.IN)
    GPIO.setup(lock_out, GPIO.OUT)
    GPIO.output(lock_out, 0)

    ir_checker = IrAlcoStatus()

    display_size = [800, 600]
    try:
        display = str(subprocess.check_output("xrandr", shell=True))
    except:
        display_status = False
    else:
        display_status = True
        display_size_str = display[display.find('current') + 8: display.find('maximum') - 2]
        display_size = [int(i) for i in display_size_str.split('x')]

    logging.info('DISPLAY {}, SIZE: {}x{}'.format(display_status, display_size[0], display_size[1]))

    frame_w = 800
    frame_h = 600

    stream = start_stream(-1, frame_w, frame_h)

    cut_display = True
    if display_size[0] > frame_w or display_size[1] > frame_h:
        cut_display = False
        display_size = [frame_w, frame_h]

    if cut_display:
        cv2.namedWindow("stream", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    face_detector = BlazeFaceDetector(os.path.join(_MODELS, _FACE_DETECTOR),
                                      os.path.join(_MODELS, _ANCHORS_NAME))

    fontpath = "Helvetica.ttc"
    ft = cv2.freetype.createFreeType2()
    ft.loadFontData(fontpath, 0)

    if cut_display:
        w_cut = (frame_w - display_size[0]) / 2
        h_cut = (frame_h - display_size[1]) / 2
        cut_display = (int(h_cut),
                       int(frame_h - h_cut),
                       int(w_cut),
                       int(frame_w - w_cut))

    if cut_display:
        # text board - 100
        # status board - 50
        percent_size = 60
        centr = [(display_size[0]) // 2, (display_size[1] - 100) // 2]
        # contour_w = int((display_size[0] - 50) / 100 * percent_size)
        contour_h = int((display_size[1] - 100) / 100 * percent_size)
        from_point_contour = [centr[0] - (contour_h // 2), centr[1] - (contour_h // 2)]
        to_point_contour = [centr[0] + (contour_h // 2), centr[1] + (contour_h // 2)]
        bbox_frame = [from_point_contour[0], from_point_contour[1], to_point_contour[0], to_point_contour[1]]
    else:
        bbox_frame = [200, 200, 450, 450]

    screen_status = 'on'
    new_person = True
    person_lost = 0
    frame_count_match = 0
    vehicle_state.set_param('engine', False)
    fps = 0.0
    t1_fps = time.monotonic()
    lost_cam = 0
    face_on_frame = False
    if cut_display:
        white_frame = cv2.imread('screen.jpg')
    else:
        white_frame = np.full(shape=[display_size[0], display_size[1], 3], fill_value=255, dtype=np.uint8)
    while True:
        vehicle_state.set_param('moving', GPIO.input(lock_in))

        if vehicle_state.get_param('auth') and vehicle_state.get_param('drunk') is None:
            if vehicle_state.get_param('drop_ir_history'):
                ir_checker.clean_test_history()
                vehicle_state.set_param('drop_ir_history', False)

            ir_checker.read_status()

            if not vehicle_state.get_param('alco_test_start'):
                vehicle_state.set_param('allowed_alco_test_run', ir_checker.allowed_alco_test_run())
            else:
                vehicle_state.set_param('alco_test_is_good', ir_checker.alco_test_is_good())

            ir_checker.alco_test_started = vehicle_state.get_param('alco_test_start')

            if ir_checker.alco_confirmed:
                vehicle_state.set_param('color_board', (0, 165, 255))
            else:
                vehicle_state.set_param('color_board', (0, 255, 255))

        if stream.isOpened():
            retval, orig_img = stream.read()
        else:
            retval = False

        subtitles = []
        if retval:
            lost_cam = 0
            orig_img = cv2.flip(orig_img, 1)
            if cut_display:
                orig_img = orig_img[cut_display[0]:cut_display[1], cut_display[2]:cut_display[3]]

            image_rgb = orig_img[..., ::-1].copy()
            if vehicle_state.get_param('auth') is None:
                # cv2.rectangle(orig_img, (bbox_frame[0], bbox_frame[1]), (bbox_frame[2], bbox_frame[3]), (0, 255, 0))
                draw_frame(orig_img, bbox_frame)

            face_detector_result = face_detector.predict_on_image(image_rgb)

            if face_detector_result:
                face_on_frame = True
                if person_lost != 0:
                    person_lost = 0
            else:
                face_on_frame = False

            inside_face = True
            if vehicle_state.get_param('auth') is None:
                inside_face = False
                for current_face in face_detector_result:
                    current_bbox = current_face['bbox']
                    if calc_iou(bbox_frame, current_bbox) > 0.8:
                        inside_face = True
            if face_detector_result and inside_face:
                if new_person:
                    new_person = False
                    vehicle_state.set_param('engine', True)
                else:
                    bbox = face_detector_result[0]['bbox']
                    face_landmark = matcher.landmark_face(image_rgb, bbox)

                    if vehicle_state.get_param('auth') is None:
                        if not matcher.get_work_status() and serverSandler.get_gpsp_status():
                            frame_count_match += 1
                            if frame_count_match > 40:
                                matcher.push_face(orig_img.copy(), face_landmark)

                    for point in face_landmark.parts():
                        cv2.circle(orig_img, (point.x, point.y), 2, (0, 0, 250), -1)

                    cv2.rectangle(orig_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 3)

                if person_lost != 0:
                    person_lost = 0

            else:
                person_lost += 1
                if (person_lost % 70) == 0 and not new_person:
                    new_person = True
                    frame_count_match = 0
                    vehicle_state.drop_state()
                    vehicle_state.set_param('engine', False)

                    if matcher.get_work_status():
                        matcher.drop_result()

                if not vehicle_state.get_param('moving') and GPIO.input(lock_out) and person_lost > 70:
                    logging.info('SET PIN LOCK_OUT 0 (LOST PERSON)')
                    GPIO.output(lock_out, 0)

            fps = (fps + (1. / (time.monotonic() - t1_fps))) / 2
            t1_fps = time.monotonic()

            cv2.putText(orig_img, "FPS: {:.2f}".format(fps), (45, 60), 0, 0.8, (0, 0, 0), 2)

            cv2.putText(orig_img, 'GPS:', (45, 90), 0, 0.8, (0, 0, 0), 2)
            if serverSandler.get_gpsp_status():
                gps_stat_color = (0, 255, 0)
            else:
                gps_stat_color = (0, 0, 255)

            cv2.circle(orig_img, (125, 80), 14, gps_stat_color, -1)

            cv2.rectangle(orig_img, (0, 0), (display_size[0], display_size[1]),
                          vehicle_state.get_param('color_board'), 70)

            cv2.rectangle(orig_img, (0, display_size[1] - 100),
                          (display_size[0], display_size[1]),
                          (255, 255, 255), -1)

            subtitles = generate_subtitles(vehicle_state, face_on_frame)

            for num, text_line in enumerate(subtitles):
                textSize = ft.getTextSize(text_line, 33, -1)
                w_text, h_text = textSize[0]
                x_text = (display_size[0] - w_text) // 2
                y_text = (display_size[1] - 100) + (45 * num)
                ft.putText(img=orig_img,
                           text=text_line,
                           org=(x_text, y_text),
                           fontHeight=33,
                           color=(0, 0, 0),
                           thickness=-1,
                           line_type=cv2.LINE_AA,
                           bottomLeftOrigin=False)

            for line in subtitles:
                if line == 'Приятного пути!':
                    # fix lost color
                    if vehicle_state.get_param('color_board') != (0, 255, 0):
                        vehicle_state.set_param('color_board', (0, 255, 0))
                elif line == 'Идентификация не пройдена':
                    if vehicle_state.get_param('color_board') != (0, 0, 255):
                        vehicle_state.set_param('color_board', (0, 0, 255))

                if line == 'Приятного пути!' and vehicle_state.get_param('moving'):
                    if screen_status != 'off':
                        logging.info('SCREEN OFF by moving status')
                        screen_off()
                        screen_status = 'off'
                    break
                elif line == 'Здравствуйте':
                    if screen_status != 'on':
                        logging.info('SCREEN ON by hi')
                        screen_on()
                        screen_status = 'on'
                    break

            if person_lost > 200 and not face_on_frame:
                if screen_status != 'on':
                    logging.info('SCREEN ON by person_lost and not face_on_frame')
                    screen_on()
                    screen_status = 'on'
            # elif face_on_frame:
            #     if screen_status != 'on':
            #         logging.info('SCREEN ON by face_on_frame')
            #         screen_on()
            #         screen_status = 'on'

            if display_status:
                cv2.imshow('stream', orig_img)
                key = cv2.waitKey(1)
                if key == 27:
                    break

            # update time when alco is running
            if matcher.get_work_status():
                vehicle_state.set_param("time", datetime.datetime.utcnow().isoformat("T") + "Z")

        else:
            lost_cam += 1
            if not new_person:
                new_person = True
                vehicle_state.drop_state()
                frame_count_match = 0
                vehicle_state.set_param('engine', False)
                stream.release()

            if not vehicle_state.get_param('moving') and GPIO.input(lock_out) and lost_cam > 500:
                logging.info('SET PIN LOCK_OUT 0  (LOST CAMERA)')
                GPIO.output(lock_out, 0)

            if os.path.exists('/dev/video1'):
                stream = start_stream(-1, frame_w, frame_h)

            # show white frame
            if display_status:
                cv2.imshow('stream', white_frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break

            continue

    logging.info('END')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='включает стартовую позицию для gps')
    args = parser.parse_args()
    start_position = False
    if args.demo:
        start_position = True

    # create logger
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] (%(processName)s - %(threadName)s: %(funcName)s) %(message)s')

    logging.info('INIT')

    GPIO.cleanup()
    GPIO.setup(37, GPIO.OUT)
    GPIO.output(37, 0)

    rs_lock = threading.Lock()
    vehicle_state = VehicleState()

    # init matcher thread
    matcher = Matcher(_KNOWNPERSON, vehicle_state, rs_lock)
    matcher.start()

    with open('car_id.txt', 'r') as f:
        car_id = f.read()
    # init server sandler thread
    serverSandler = ServerSandler(logging, car_id, 'http://49.12.154.228:8000', rs_lock, vehicle_state, start_position)
    serverSandler.start()

    main(vehicle_state, serverSandler, matcher)

    serverSandler.stop()
    serverSandler.join()

    matcher.stop()
    matcher.join()

    logging.info('END')