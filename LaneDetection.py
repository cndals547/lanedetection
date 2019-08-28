import cv2 # OpenCV를 사용하기 위한 cv2 임포트
import numpy as np # Numpy를 np이름으로 임포트
from collections import deque # collections 모듈에서 제공하는 자료구조인 deque을 사용하기 위한 임포트

def lane_detection(image):

    def convert_hsv(image): # BGR이미지 -> HSV이미지 변환 함수
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def convert_hls(image): # BGR이미지 -> HLS이미지 변환 함수
        return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    def convert_gray(image): # 컬러이미지 -> 흑백이미지 변환 함수
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 가우시안 필터 함수(커널 크기 : 9, X-축 방향으로의 가우시안 커널 표준편차(SigmaX) : 0)
    def applied_Gaussian(image, kerner_size = 15):
        return cv2.GaussianBlur(image, (kerner_size, kerner_size), 0)

    # Canny 에지 검출 함수(낮은 값의 임계값 : 50, 높은 값의 임계값 : 150)
    def detected_Canny(image, low_threshold=50, high_threshold=150):
        return cv2.Canny(image, low_threshold, high_threshold)

    def White_Yellow_Detection(image): # 이미지에서 흰색, 노란색 검출 함수
        converted = convert_hls(image) # BGR이미지를 HLS 이미지로 변환
        #lower = np.array([0, 200, 0], dtype=np.uint8)
        #upper = np.array([255, 255, 255], dtype=np.uint8)
        lower = np.uint8([0, 200, 0]) # 최소 흰색 범위 정의
        upper = np.uint8([255, 255, 255]) # 최대 흰색 범위 정의
        white_mask = cv2.inRange(converted, lower, upper) # 이미지에서 흰색만 추출하기 위한 임계값

        #lower = np.array([10, 0, 40], dtype=np.uint8)
        #upper = np.array([40, 255, 255], dtype=np.uint8)
        lower = np.uint8([10, 0, 100]) # 최소 노란색 범위 정의
        upper = np.uint8([40, 255, 255]) # 최대 노란색 범위 정의
        yellow_mask = cv2.inRange(converted, lower, upper) # 이미지에서 노란색만 추출하기 위한 임계값

        mask = cv2.bitwise_or(white_mask, yellow_mask) # white_mask와 yellow_mask이미지 OR 연산
        applied_mask = cv2.bitwise_and(image, image, mask = mask) # 이미지에 mask 적용
        return applied_mask

    def filter_region(image, vertices): # 검출된 ROI 영역 color로 채우는 함수
        mask = np.zeros_like(image) # image 크기만큼의 영행렬을 mask 변수에 저장
        if len(mask.shape) == 2: # mask이미지가 3채널 이라면 :
            cv2.fillPoly(mask, vertices, 255) # vertices 영역만큼의 다각형부분을 (255)값으로 채움
        else: # mask이미지가 1채널 이라면 :
            cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # vertices 영역만큼의 다각형부분을 (255, 255, 255)값으로 채움
        return cv2.bitwise_and(image, mask)

    def region_of_interest(image): # 관심영역 설정 함수
        height, width = image.shape[:2] # 이미지의 높이와 너비
        bottom_left = [width * 0.1, height * 0.95] # bottom_left(왼쪽 하단)의 높이와 너비
        top_left = [width * 0.4, height * 0.6] # top_left(왼쪽 상단)의 높이와 너비
        bottom_right = [width * 0.9, height * 0.95] # bottom_right(오른쪽 하단)의 높이와 너비
        top_right = [width * 0.6, height * 0.6] # top_right(오른쪽 상단)의 높이와 너비
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32) # 관심영역인 다각형의 꼭지점 정의
        return filter_region(image, vertices)

    # 허프 변환 함수(원점으로부터의 거리 간격(rho) : 1, x축과의 각도로 라디안 간격(theta) : np.pi/180,
    #                직선을 검출하기 위한 어큐뮬레이터의 임계값(threshold) : 20,
    #                검출할 최소 직선의 길이(minLineLength) : 20, 직선 위의 에지들의 최대 허용 간격(maxLineGap) : 30)
    def hough_transform(image):
        return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=30)

    def make_line_points(image, lines_parameters):
        try:
            slope, intercept = lines_parameters
        except TypeError:
            slope, intercept = 0, 0
        y1 = int(image.shape[0])
        y2 = int(y1 * 0.6)
        try:
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
        except ZeroDivisionError:
            x1, x2 = 0, 0
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(image, lines): # 기울기가 유사한 직선들의 값들을 평균내서 반환하는 함수
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_line_points(image, left_fit_average)
        right_line = make_line_points(image, right_fit_average)
        return np.array([left_line, right_line])

    def draw_lane_lines(image, lines, color=[0, 0, 255], thickness=5): # 영상에서 라인(차선)을 인식하고 그려주는 함수
        line_image = np.zeros_like(image)
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
        return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0) # image와 line_image 겹침

    def center_line(image, lines):
        line_image = np.zeros_like(image)
        x1_1, y1_1, x2_1, y2_1 = lines[0].reshape(4)
        x1_2, y1_2, x2_2, y2_2 = lines[1].reshape(4)
        ym1 = y1_1
        ym2 = y2_1
        xm1 = int(x1_1 - (x1_1 - x1_2) / 2)
        xm2 = int(x2_1 - (x2_1 - x2_2) / 2)
        cv2.line(line_image,
                 (xm1, ym1),
                 (xm2, ym2),
                 color=[0, 255, 0],
                 thickness=10)
        return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0) # image와 line_image 겹침


    lane_image = np.copy(image)
    white_yellow_frame = White_Yellow_Detection(lane_image)  # 이미지에서 흰색, 노란색 검출
    Gaussian_frame = applied_Gaussian(white_yellow_frame)  # 가우시안 필터 적용
    Canny_frame = detected_Canny(Gaussian_frame)  # Canny 에지 검출 알고리즘 적용
    roi_frame = region_of_interest(Canny_frame)  # 관심영역 설정
    hough_frame = hough_transform(roi_frame)  # 허프 변환 적용
    average_line_frame = average_slope_intercept(frame, hough_frame)
    center_image = center_line(lane_image, average_line_frame)
    lane_image = draw_lane_lines(center_image, average_line_frame)
    return lane_image

cap = cv2.VideoCapture("C:/Users/cndal/OneDrive/바탕 화면/test_videos/solidYellowLeft.mp4") # 영상 입력

while (cap.isOpened()):
    ret, frame = cap.read()
    result = lane_detection(frame)
    # white_yellow_frame = White_Yellow_Detection(frame) # 이미지에서 흰색, 노란색 검출
    # Gaussian_frame = applied_Gaussian(white_yellow_frame) # 가우시안 필터 적용
    # Canny_frame = detected_Canny(Gaussian_frame) # Canny 에지 검출 알고리즘 적용
    # roi_frame = region_of_interest(Canny_frame) #관심영역 설정
    # hough_frame = hough_transform(roi_frame) # 허프 변환 적용
    # average_line_frame = lane_lines(frame, hough_frame)
    # center_line_frame = center_line(frame, average_line_frame)
    # result = draw_lane_lines(center_line_frame, average_line_frame) # 차선 그리기
    cv2.imshow('Original', frame) # 원본 영상 출력
    cv2.imshow('result', result) # 결과 영상 출력

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
